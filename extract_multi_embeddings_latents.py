from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse
import torch

from interplm.embedders.esm import ESM
from interplm.sae.inference import load_sae_from_hf

from variables import address_dict, subfolders
from utils import (
    fetch_sequences_from_fasta,
    get_best_device,
    safe_torch_save,
    safe_numpy_save,
    initialize_batch_metadata_csv,
    update_batch_metadata_flags,
    get_batch_progress,
    validate_metadata_csv,
    chunked,
)


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "cuda out of memory" in msg


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_esm_model(
    model_name: str,
    max_length: int,
) -> ESM:
    return ESM(model_name=model_name, max_length=max_length)


def load_sae_models(
    plm_model: str,
    plm_layer_list: List[int],
    device: str,
) -> Dict[int, torch.nn.Module]:
    sae_models: Dict[int, torch.nn.Module] = {}
    for layer in plm_layer_list:
        sae = load_sae_from_hf(plm_model=plm_model, plm_layer=layer).to(device)
        sae.eval()
        sae_models[layer] = sae
        _clear_memory()
    return sae_models


def extract_esm_embeddings(
    esm: ESM,
    sequences: List[str],
    layers: List[int],
    batch_size: int = 8,
) -> Dict[int, torch.Tensor]:
    """
    Returns CPU tensors:
      Dict[layer] -> tensor of shape (total_tokens_in_batch, d_model)
    """
    current_batch_size = max(1, int(batch_size))

    while True:
        try:
            with torch.inference_mode():
                embeddings = esm.extract_embeddings_multiple_layers(
                    sequences=sequences,
                    layers=layers,
                    batch_size=current_batch_size,
                )

            # Force all outputs to CPU and detach from any graph/state.
            output = {
                layer: embeddings[layer].detach().cpu().contiguous()
                for layer in layers
            }
            del embeddings
            _clear_memory()
            return output

        except Exception as exc:
            if not _is_cuda_oom(exc) or current_batch_size == 1:
                raise
            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"OOM during ESM extraction with plm_batch_size={current_batch_size}. "
                f"Retrying with plm_batch_size={next_batch_size}."
            )
            current_batch_size = next_batch_size
            _clear_memory()


def save_batch_embeddings(
    batch_embeddings: Dict[int, torch.Tensor],
    embeddings_dir: Path,
    batch_index: int,
    plm_layer_list: List[int],
) -> None:
    _ensure_dir(embeddings_dir)
    for plm_layer in plm_layer_list:
        emb_path = embeddings_dir / f"batch_{batch_index:06d}-layer_{plm_layer}.pt"
        safe_torch_save(batch_embeddings[plm_layer], emb_path)
        print(
            f"Saved batch embeddings (batch {batch_index}, layer {plm_layer}): "
            f"{tuple(batch_embeddings[plm_layer].shape)} {emb_path}"
        )


def load_batch_embeddings(
    embeddings_dir: Path,
    batch_index: int,
    plm_layer_list: List[int],
) -> Dict[int, torch.Tensor]:
    batch_embeddings: Dict[int, torch.Tensor] = {}
    for plm_layer in plm_layer_list:
        emb_path = embeddings_dir / f"batch_{batch_index:06d}-layer_{plm_layer}.pt"
        batch_embeddings[plm_layer] = torch.load(emb_path, map_location="cpu")
    return batch_embeddings


def encode_sae_latents_for_layer(
    embeddings_cpu: torch.Tensor,
    sae: torch.nn.Module,
    device: str,
    row_batch_size: int,
    batch_index: int,
    plm_layer: int,
) -> scipy.sparse.csr_matrix:
    """
    Encodes one layer's embeddings into one batch-concatenated sparse CSR matrix.
    Keeps only sparse chunks in memory.
    """
    current_row_batch_size = max(1, int(row_batch_size))
    sparse_chunks: List[scipy.sparse.csr_matrix] = []

    start = 0
    with torch.inference_mode():
        while start < embeddings_cpu.shape[0]:
            end = min(start + current_row_batch_size, embeddings_cpu.shape[0])
            emb_chunk_cpu = embeddings_cpu[start:end]

            try:
                emb_chunk_gpu = emb_chunk_cpu.to(device, non_blocking=True)
                latent_chunk_gpu = sae.encode(emb_chunk_gpu)

                # Immediately move off GPU
                latent_chunk_cpu = latent_chunk_gpu.detach().cpu()

                # Convert chunk to sparse immediately to reduce RAM pressure
                latent_chunk_np = latent_chunk_cpu.numpy()
                latent_chunk_sp = scipy.sparse.csr_matrix(latent_chunk_np)
                sparse_chunks.append(latent_chunk_sp)

                del emb_chunk_gpu, latent_chunk_gpu, latent_chunk_cpu, latent_chunk_np, latent_chunk_sp
                start = end

            except Exception as exc:
                if not _is_cuda_oom(exc) or current_row_batch_size == 1:
                    raise
                next_row_batch_size = max(1, current_row_batch_size // 2)
                print(
                    f"OOM during SAE encoding (batch {batch_index}, layer {plm_layer}) "
                    f"with row_batch_size={current_row_batch_size}. "
                    f"Retrying with row_batch_size={next_row_batch_size}."
                )
                current_row_batch_size = next_row_batch_size
                _clear_memory()

            _clear_memory()

    if not sparse_chunks:
        # Should not happen for non-empty embeddings, but keep safe.
        return scipy.sparse.csr_matrix((0, 0))

    latents_sparse = scipy.sparse.vstack(sparse_chunks, format="csr")

    # release chunk list early
    del sparse_chunks
    _clear_memory()

    return latents_sparse


def save_batch_latents(
    batch_embeddings: Dict[int, torch.Tensor],
    sae_models: Dict[int, torch.nn.Module],
    latents_dir: Path,
    batch_index: int,
    plm_layer_list: List[int],
    device: str,
    sae_row_batch_size: int = 32768,
) -> None:
    _ensure_dir(latents_dir)

    for plm_layer in plm_layer_list:
        embeddings_cpu = batch_embeddings[plm_layer]
        sae = sae_models[plm_layer]

        latents_sparse = encode_sae_latents_for_layer(
            embeddings_cpu=embeddings_cpu,
            sae=sae,
            device=device,
            row_batch_size=sae_row_batch_size,
            batch_index=batch_index,
            plm_layer=plm_layer,
        )

        latent_path = latents_dir / f"batch_{batch_index:06d}-layer_{plm_layer}.npz"
        safe_numpy_save(latents_sparse, latent_path)
        print(
            f"Saved batch latents (batch {batch_index}, layer {plm_layer}): "
            f"{latents_sparse.shape} {latent_path}"
        )

        del latents_sparse
        _clear_memory()


if __name__ == "__main__":
    # ===== 1) Run Configuration =====
    data_folder = address_dict["plm-interpret-data-ssd"]
    data_subfolder = "uniprot_sprot90"
    fasta_fname = "uniprot_sprot90.fasta"

    fasta_fpath = (
        Path(data_folder)
        / subfolders["sequences"]
        / data_subfolder
        / fasta_fname
    )

    model_name = "facebook/esm2_t33_650M_UR50D"
    plm_model = "esm2-650m"
    plm_layer_list = [9, 18, 24, 30, 33]

    plm_batch_size = 8
    seq_batch_size = 1000
    sae_row_batch_size = 32768
    max_length = 1536
    start_from_batch_index = 100

    embeddings_dir = (
        Path(data_folder)
        / subfolders["protein_embeddings"]
        / data_subfolder
    )
    latents_dir = (
        Path(data_folder)
        / subfolders["sae_latents"]
        / data_subfolder
    )
    metadata_csv_fpath = (
        Path(data_folder)
        / subfolders["protein_embeddings"]
        / data_subfolder
        / "batch_sequence_index.csv"
    )

    _ensure_dir(embeddings_dir)
    _ensure_dir(latents_dir)

    # ===== 2) Device + Runtime Setup =====
    device = get_best_device()
    device_str = str(device)
    print("Device:", device_str)
    if device_str == "cuda":
        print(
            "Tip: if fragmentation persists, run with "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        )

    # ===== 3) Input Loading =====
    sequences, seq_names, _ = fetch_sequences_from_fasta(str(fasta_fpath))
    print(f"{len(sequences)} sequences total.")

    # ===== 4) Metadata CSV Initialization / Resume Validation =====
    if metadata_csv_fpath.exists():
        print(f"Metadata CSV exists, resuming from: {metadata_csv_fpath}")
        validate_metadata_csv(
            sequences=sequences,
            seq_names=seq_names,
            seq_batch_size=seq_batch_size,
            max_length=max_length,
            csv_fpath=str(metadata_csv_fpath),
        )
        print("Metadata CSV validation passed.")
    else:
        initialize_batch_metadata_csv(
            sequences=sequences,
            seq_names=seq_names,
            seq_batch_size=seq_batch_size,
            max_length=max_length,
            csv_fpath=str(metadata_csv_fpath),
        )
        print(f"Metadata CSV created: {metadata_csv_fpath}")

    # ===== 5) Model Initialization =====
    print("Loading ESM model once...")
    esm = load_esm_model(
        model_name=model_name,
        max_length=max_length,
    )
    _clear_memory()

    print("Loading SAE models once...")
    sae_models = load_sae_models(
        plm_model=plm_model,
        plm_layer_list=plm_layer_list,
        device=device_str,
    )
    _clear_memory()

    try:
        # ===== 6) Batch Processing =====
        for batch_start, batch_seqs in chunked(sequences, seq_batch_size):

            batch_index = batch_start // seq_batch_size
            processed_seq_indices = range(batch_start, batch_start + len(batch_seqs))
            if batch_index < start_from_batch_index:
                continue

            print(
                f"\nBatch {batch_index} "
                f"({batch_start}–{batch_start + len(batch_seqs) - 1}, {len(batch_seqs)} seqs)"
            )

            emb_paths = [
                embeddings_dir / f"batch_{batch_index:06d}-layer_{plm_layer}.pt"
                for plm_layer in plm_layer_list
            ]
            latent_paths = [
                latents_dir / f"batch_{batch_index:06d}-layer_{plm_layer}.npz"
                for plm_layer in plm_layer_list
            ]

            progress = get_batch_progress(str(metadata_csv_fpath), processed_seq_indices)
            embeddings_done = progress["all_embedding_obtained"] and all(p.exists() for p in emb_paths)
            latents_done = progress["all_latent_obtained"] and all(p.exists() for p in latent_paths)

            if latents_done:
                print(f"Batch {batch_index}: embeddings+latents already complete, skipping.")
                continue

            if embeddings_done:
                print(
                    f"Batch {batch_index}: embeddings already complete, "
                    f"loading from disk for latent extraction."
                )
                batch_embeddings = load_batch_embeddings(
                    embeddings_dir=embeddings_dir,
                    batch_index=batch_index,
                    plm_layer_list=plm_layer_list,
                )
            else:
                batch_embeddings = extract_esm_embeddings(
                    esm=esm,
                    sequences=batch_seqs,
                    layers=plm_layer_list,
                    batch_size=plm_batch_size,
                )
                save_batch_embeddings(
                    batch_embeddings=batch_embeddings,
                    embeddings_dir=embeddings_dir,
                    batch_index=batch_index,
                    plm_layer_list=plm_layer_list,
                )
                update_batch_metadata_flags(
                    str(metadata_csv_fpath),
                    processed_seq_indices=processed_seq_indices,
                    embedding_obtained=True,
                )

            save_batch_latents(
                batch_embeddings=batch_embeddings,
                sae_models=sae_models,
                latents_dir=latents_dir,
                batch_index=batch_index,
                plm_layer_list=plm_layer_list,
                device=device_str,
                sae_row_batch_size=sae_row_batch_size,
            )
            update_batch_metadata_flags(
                str(metadata_csv_fpath),
                processed_seq_indices=processed_seq_indices,
                latent_obtained=True,
            )

            del batch_embeddings
            _clear_memory()

            if device_str == "cuda":
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"CUDA memory after batch {batch_index}: "
                    f"allocated={allocated_gb:.2f} GB, reserved={reserved_gb:.2f} GB"
                )

    finally:
        # ===== 7) Cleanup =====
        del sae_models
        del esm
        _clear_memory()
