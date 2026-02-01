from interplm.sae.inference import load_sae_from_hf
from interplm.embedders.esm import ESM
import torch
import scipy
from typing import List
from variables import address_dict, subfolders
from utils import fetch_sequences_from_fasta

def get_esm_embeddings(
        sequences,
        seq_names,
        model_name,
        layers,
        batch_size=8,
        max_length=2048,
        embeddings_dir=None,
        device='cpu'
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
                print(f'[{i}] Saved embeddings (layer {layer}): {len(seq)} {emb_lyr.shape} {emb_fpath_torch}')
                # update seq_start_idx
                seq_start_idx = seq_end_idx
    return embeddings

def get_sae_latents(seq_names, embeddings_dir, latents_dir, plm_layer,  plm_model='esm2-650m', device='cpu'):
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
        print(f'[{i}] Saved latents (layer {plm_layer}): {latent_sparse_fpath}')


if __name__=='__main__':
    data_folder = address_dict['databases']
    data_subfolder = 'uniprot_sprot/raw/'
    fasta_fname = 'uniprot_sprot_SNIPPET.fasta'
    fasta_fpath = f'{data_folder}{data_subfolder}{fasta_fname}'
    model_name = "facebook/esm2_t33_650M_UR50D"
    plm_model = "esm2-650m"
    plm_layer_list = [9, 18, 24, 30, 33]  # Choose ESM layer (1,9,18,24,30,33)
    batch_size = 8
    max_length = 1536
    embeddings_dir = f"{address_dict['plm-interpret-data']}{subfolders['protein_embeddings']}/uniprot_sprot_SNIPPET/"
    latents_dir = f"{address_dict['plm-interpret-data']}{subfolders['sae_latents']}/uniprot_sprot_SNIPPET/"
    # get sequences
    sequences, seq_names, _ = fetch_sequences_from_fasta(fasta_fpath)
    print(f'Processing {len(sequences)} sequences...')

    # get device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # get ESM embeddings
    embeddings = get_esm_embeddings(
        sequences,
        seq_names,
        model_name,
        plm_layer_list,
        batch_size=batch_size,
        max_length=max_length,
        embeddings_dir=embeddings_dir,
        device=str(device)
    )

    # get SAE latents
    for plm_layer in plm_layer_list:
        get_sae_latents(
            seq_names,
            embeddings_dir,
            latents_dir,
            plm_layer,
            plm_model='esm2-650m',
            device=str(device)
        )