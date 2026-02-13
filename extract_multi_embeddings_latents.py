from interplm.sae.inference import load_sae_from_hf
from interplm.embedders.esm import ESM
import torch
import scipy
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
    chunked,
)

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
    embeddings = esm.extract_embeddings_multiple_layers(sequences=sequences, layers=layers, batch_size=batch_size)
    return {layer: embeddings[layer].to(device) for layer in layers}

def get_sae_latents(
        batch_embeddings: Dict[int, torch.Tensor],
        latents_dir,
        batch_index,
        plm_layer_list,
        plm_model='esm2-650m',
        device='cpu',
):
    """
    Load SAE models by layer and save batch-concatenated sparse latents.
    """
    for plm_layer in plm_layer_list:
        sae = load_sae_from_hf(plm_model=plm_model, plm_layer=plm_layer).to(device)
        latents = sae.encode(batch_embeddings[plm_layer])
        latents = latents.cpu().detach().numpy()
        latents_sparse = scipy.sparse.csr_matrix(latents)
        latent_batch_fpath = f'{latents_dir}batch_{batch_index:06d}-layer_{plm_layer}.npz'
        safe_numpy_save(latents_sparse, Path(latent_batch_fpath))
        print(f'Saved batch latents (batch {batch_index}, layer {plm_layer}): {latents_sparse.shape} {latent_batch_fpath}')


if __name__=='__main__':
    data_folder = address_dict['plm-interpret-data-ssd'] # address_dict['plm-interpret-data-ssd'] #
    data_subfolder = 'uniprot_sprot90'
    fasta_fname = 'uniprot_sprot90.fasta'
    fasta_fpath = f'{data_folder}{subfolders["sequences"]}{data_subfolder}/{fasta_fname}'
    model_name = "facebook/esm2_t33_650M_UR50D"
    plm_model = "esm2-650m"
    plm_layer_list = [9, 18, 24, 30, 33]  # Choose ESM layer (1,9,18,24,30,33)
    plm_batch_size = 8
    seq_batch_size = 1000
    max_length = 1536
    embeddings_dir = f"{data_folder}{subfolders['protein_embeddings']}{data_subfolder}/"
    latents_dir = f"{data_folder}{subfolders['sae_latents']}{data_subfolder}/"
    metadata_csv_fpath = f"{data_folder}{subfolders['protein_embeddings']}{data_subfolder}/batch_sequence_index.csv"

    # get device
    device = get_best_device()
    device_str = str(device)
    print('Device:', device_str)

    # get sequences
    sequences, seq_names, _ = fetch_sequences_from_fasta(fasta_fpath)
    print(f'{len(sequences)} sequences total.')
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
        )
        update_batch_metadata_flags(
            metadata_csv_fpath,
            processed_seq_indices=processed_seq_indices,
            latent_obtained=True
        )
