from interplm.sae.inference import load_sae_from_hf
from interplm.embedders.esm import ESM
import torch
import scipy
import numpy as np
import gc
from pathlib import Path
from typing import Dict
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

def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "cuda out of memory" in msg


def get_esm_embeddings(
        sequences,
        model_name,
        layers,
        batch_size=8,
        max_length=2048,
        device='cpu',
):
    """
    Extract ESM2 embeddings from sequences
    :param model_name:
    :param sequences:
    :param layers:
    :param batch_size:
    :param embeddings_dir:
    :return:
    """
    # batch extract embeddings
    esm = ESM(model_name=model_name, max_length=max_length)
    current_batch_size = max(1, int(batch_size))
    while True:
        try:
            with torch.inference_mode():
                embeddings = esm.extract_embeddings_multiple_layers(
                    sequences=sequences,
                    layers=layers,
                    batch_size=current_batch_size
                )
            # Keep concatenated embeddings on CPU to avoid GPU memory pressure downstream.
            output = {layer: embeddings[layer].detach().cpu() for layer in layers}
            del embeddings
            _clear_cuda_cache()
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
            _clear_cuda_cache()
            gc.collect()

def get_sae_latents(
        batch_embeddings: Dict[int, torch.Tensor],
        latents_dir,
        batch_index,
        plm_layer_list,
        plm_model='esm2-650m',
        device='cpu',
        sae_row_batch_size=32768,
):
    """
    Load SAE models by layer and save batch-concatenated sparse latents.
    """
    for plm_layer in plm_layer_list:
        sae = load_sae_from_hf(plm_model=plm_model, plm_layer=plm_layer).to(device)
        embeddings_cpu = batch_embeddings[plm_layer]
        row_batch_size = max(1, int(sae_row_batch_size))
        start = 0
        latent_chunks = []
        try:
            with torch.inference_mode():
                while start < embeddings_cpu.shape[0]:
                    end = min(start + row_batch_size, embeddings_cpu.shape[0])
                    emb_chunk = embeddings_cpu[start:end]
                    try:
                        emb_chunk = emb_chunk.to(device, non_blocking=True)
                        latent_chunk = sae.encode(emb_chunk)
                        latent_chunks.append(latent_chunk.detach().cpu().numpy())
                        del emb_chunk, latent_chunk
                        start = end
                    except Exception as exc:
                        if not _is_cuda_oom(exc) or row_batch_size == 1:
                            raise
                        next_row_batch_size = max(1, row_batch_size // 2)
                        print(
                            f"OOM during SAE encoding (batch {batch_index}, layer {plm_layer}) "
                            f"with row_batch_size={row_batch_size}. Retrying with row_batch_size={next_row_batch_size}."
                        )
                        row_batch_size = next_row_batch_size
                        _clear_cuda_cache()
                        gc.collect()

            latents = np.concatenate(latent_chunks, axis=0)
            latents_sparse = scipy.sparse.csr_matrix(latents)
            latent_batch_fpath = f'{latents_dir}batch_{batch_index:06d}-layer_{plm_layer}.npz'
            safe_numpy_save(latents_sparse, Path(latent_batch_fpath))
            print(f'Saved batch latents (batch {batch_index}, layer {plm_layer}): {latents_sparse.shape} {latent_batch_fpath}')
        finally:
            del sae
            _clear_cuda_cache()
            gc.collect()

def load_batch_embeddings(
        embeddings_dir,
        batch_index,
        plm_layer_list,
):
    batch_embeddings = {}
    for plm_layer in plm_layer_list:
        emb_batch_fpath = f'{embeddings_dir}batch_{batch_index:06d}-layer_{plm_layer}.pt'
        batch_embeddings[plm_layer] = torch.load(emb_batch_fpath, map_location="cpu")
    return batch_embeddings


if __name__=='__main__':
    data_folder = address_dict['plm-interpret-data-ssd'] # address_dict['plm-interpret-data-ssd'] #
    data_subfolder = 'uniprot_sprot80'
    fasta_fname = 'uniprot_sprot80.fasta'
    fasta_fpath = f'{data_folder}{subfolders["sequences"]}{data_subfolder}/{fasta_fname}'
    model_name = "facebook/esm2_t33_650M_UR50D"
    plm_model = "esm2-650m"
    plm_layer_list = [9, 18, 24, 30, 33]  # Choose ESM layer (1,9,18,24,30,33)
    plm_batch_size = 8
    seq_batch_size = 1000
    sae_row_batch_size = 32768
    max_length = 1536
    embeddings_dir = f"{data_folder}{subfolders['protein_embeddings']}{data_subfolder}/"
    latents_dir = f"{data_folder}{subfolders['sae_latents']}{data_subfolder}/"
    metadata_csv_fpath = f"{data_folder}{subfolders['protein_embeddings']}{data_subfolder}/batch_sequence_index.csv"

    # get device
    device = get_best_device()
    device_str = str(device)
    print('Device:', device_str)
    if device_str == "cuda":
        print("Tip: if fragmentation persists, run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # get sequences
    sequences, seq_names, _ = fetch_sequences_from_fasta(fasta_fpath)
    print(f'{len(sequences)} sequences total.')
    if Path(metadata_csv_fpath).exists():
        print(f'Metadata CSV exists, resuming from: {metadata_csv_fpath}')
        validate_metadata_csv(
            sequences=sequences,
            seq_names=seq_names,
            seq_batch_size=seq_batch_size,
            max_length=max_length,
            csv_fpath=metadata_csv_fpath,
        )
        print('Metadata CSV validation passed.')
    else:
        initialize_batch_metadata_csv(
            sequences=sequences,
            seq_names=seq_names,
            seq_batch_size=seq_batch_size,
            max_length=max_length,
            csv_fpath=metadata_csv_fpath
        )
        print(f'Metadata CSV created: {metadata_csv_fpath}')

    for batch_start, batch_seqs in chunked(sequences, seq_batch_size):
        batch_index = batch_start // seq_batch_size
        processed_seq_indices = range(batch_start, batch_start + len(batch_seqs))
        print(f"\nBatch {batch_index} ({batch_start}–{batch_start + len(batch_seqs) - 1}, {len(batch_seqs)} seqs)")

        emb_paths = [
            Path(f'{embeddings_dir}batch_{batch_index:06d}-layer_{plm_layer}.pt')
            for plm_layer in plm_layer_list
        ]
        latent_paths = [
            Path(f'{latents_dir}batch_{batch_index:06d}-layer_{plm_layer}.npz')
            for plm_layer in plm_layer_list
        ]
        progress = get_batch_progress(metadata_csv_fpath, processed_seq_indices)
        embeddings_done = progress["all_embedding_obtained"] and all(p.exists() for p in emb_paths)
        latents_done = progress["all_latent_obtained"] and all(p.exists() for p in latent_paths)

        if latents_done:
            print(f'Batch {batch_index}: embeddings+latents already complete, skipping.')
            continue

        if embeddings_done:
            print(f'Batch {batch_index}: embeddings already complete, loading from disk for latent extraction.')
            batch_embeddings = load_batch_embeddings(
                embeddings_dir=embeddings_dir,
                batch_index=batch_index,
                plm_layer_list=plm_layer_list,
            )
        else:
            # 1) ESM embeddings (save one concatenated tensor per layer per batch)
            batch_embeddings = get_esm_embeddings(
                sequences=batch_seqs,
                model_name=model_name,
                layers=plm_layer_list,
                batch_size=plm_batch_size,
                max_length=max_length,
                device=device_str,
            )
            for plm_layer in plm_layer_list:
                emb_batch_fpath = f'{embeddings_dir}batch_{batch_index:06d}-layer_{plm_layer}.pt'
                safe_torch_save(batch_embeddings[plm_layer].cpu(), Path(emb_batch_fpath))
                print(f'Saved batch embeddings (batch {batch_index}, layer {plm_layer}): {tuple(batch_embeddings[plm_layer].shape)} {emb_batch_fpath}')
            update_batch_metadata_flags(
                metadata_csv_fpath,
                processed_seq_indices=processed_seq_indices,
                embedding_obtained=True
            )

        # 2) SAE latents (save one concatenated sparse matrix per layer per batch)
        get_sae_latents(
            batch_embeddings=batch_embeddings,
            latents_dir=latents_dir,
            batch_index=batch_index,
            plm_layer_list=plm_layer_list,
            plm_model=plm_model,
            device=device_str,
            sae_row_batch_size=sae_row_batch_size,
        )
        update_batch_metadata_flags(
            metadata_csv_fpath,
            processed_seq_indices=processed_seq_indices,
            latent_obtained=True
        )
        del batch_embeddings
        _clear_cuda_cache()
        gc.collect()
