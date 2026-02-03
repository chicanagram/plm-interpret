from interplm.sae.inference import load_sae_from_hf
from interplm.embedders.esm import ESM
import torch
import scipy
from typing import List, Sequence, Tuple
from variables import address_dict, subfolders
from utils import fetch_sequences_from_fasta, get_best_device

def get_esm_embeddings(
        sequences,
        seq_names,
        model_name,
        layers,
        batch_size=8,
        max_length=2048,
        embeddings_dir=None,
        device='cpu',
        batch_start=0
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

    # save embeddings individually
    if embeddings_dir is not None:
        for layer in layers:
            seq_start_idx = 0
            embeddings_layer = embeddings[layer].to(device)
            for i, (seq, seq_name) in enumerate(zip(sequences, seq_names)):
                seq_end_idx = seq_start_idx + min(len(seq),max_length)
                emb_lyr = embeddings_layer[seq_start_idx:seq_end_idx,:].clone()
                emb_fpath_torch = f'{embeddings_dir}{seq_name}-{layer}.pt'
                torch.save(emb_lyr, emb_fpath_torch)
                print(f'[{i+batch_start}] Saved embeddings (layer {layer}): {len(seq)} {emb_lyr.shape} {emb_fpath_torch}')
                # update seq_start_idx
                seq_start_idx = seq_end_idx
    return embeddings

def get_sae_latents(
        seq_names,
        embeddings_dir,
        latents_dir,
        plm_layer,
        plm_model='esm2-650m',
        device='cpu',
        batch_start=0
):
    """
    Load SAE model and extract features
    :param plm_layer_list:
    :return:
    """
    sae = load_sae_from_hf(plm_model=plm_model, plm_layer=plm_layer).to(device)
    for i, seq_name in enumerate(seq_names):
        # get SAE latents
        emb_fpath = f'{embeddings_dir}{seq_name}-{plm_layer}.pt'
        embeddings = torch.load(emb_fpath)
        latents = sae.encode(embeddings)

        # convert to sparse array in numpy
        latents = latents.cpu().detach().numpy()
        latents_sparse = scipy.sparse.csr_matrix(latents)
        latent_sparse_fpath = f'{latents_dir}{seq_name}-{plm_layer}.npz'
        scipy.sparse.save_npz(latent_sparse_fpath, latents_sparse)
        print(f'[{i+batch_start}] Saved latents (layer {plm_layer}): {latent_sparse_fpath}')

def chunked(iterable: Sequence, chunk_size: int):
    for start in range(0, len(iterable), chunk_size):
        yield start, iterable[start:start + chunk_size]




if __name__=='__main__':
    data_folder = address_dict['plm-interpret-data-ssd'] # address_dict['plm-interpret-data']
    data_subfolder = 'uniprot_sprot90'
    fasta_fname = 'uniprot_sprot90.fasta'
    fasta_fpath = f'{data_folder}{subfolders["sequences"]}{data_subfolder}/{fasta_fname}'
    model_name = "facebook/esm2_t33_650M_UR50D"
    plm_model = "esm2-650m"
    plm_layer_list = [9, 18, 24, 30, 33]  # Choose ESM layer (1,9,18,24,30,33)
    plm_batch_size = 8
    seq_batch_size = 1000
    max_length = 1536
    save_idx_in_fname = False
    embeddings_dir = f"{address_dict['plm-interpret-data']}{subfolders['protein_embeddings']}{data_subfolder}/"
    latents_dir = f"{address_dict['plm-interpret-data']}{subfolders['sae_latents']}{data_subfolder}/"

    # get device
    device = get_best_device()
    device_str = str(device)
    print('Device:', device_str)

    # get sequences
    sequences, seq_names, _ = fetch_sequences_from_fasta(fasta_fpath)
    print(f'{len(sequences)} sequences total.')

    for batch_start, batch_seqs in chunked(sequences, seq_batch_size):
        batch_names_raw = seq_names[batch_start:batch_start + len(batch_seqs)]
        if save_idx_in_fname:
            batch_names = [f"{batch_start + i:08d}_{name}" for i, name in enumerate(batch_names_raw)]
        else:
            batch_names = batch_names_raw.copy()
        print(f"\nBatch {batch_start}–{batch_start + len(batch_seqs) - 1} ({len(batch_seqs)} seqs)")

        # 1) ESM embeddings (saves per-seq .pt files into embeddings_dir)
        _ = get_esm_embeddings(
            sequences=batch_seqs,
            seq_names=batch_names,
            model_name=model_name,
            layers=plm_layer_list,
            batch_size=plm_batch_size,
            max_length=max_length,
            embeddings_dir=embeddings_dir,
            device=device_str,
            batch_start=batch_start
        )

        # 2) SAE latents (reads those .pt files and writes latents)
        for plm_layer in plm_layer_list:
            get_sae_latents(
                seq_names=batch_names,
                embeddings_dir=embeddings_dir,
                latents_dir=latents_dir,
                plm_layer=plm_layer,
                plm_model="esm2-650m",
                device=device_str,
                batch_start=batch_start
            )