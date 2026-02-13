import scipy
import torch
from .batch_metadata import get_sequence_metadata_row


def parse_concatenated_representation(
        representation_fpath: str,
        csv_fpath: str,
        seq_index: int,
        sparse_format: bool = False,
):
    """
    Load a concatenated embedding/latent representation file and parse one sequence.

    - embeddings: set sparse_format=False, representation_fpath should be .pt
    - latents: set sparse_format=True, representation_fpath should be .npz
    """
    row = get_sequence_metadata_row(csv_fpath=csv_fpath, seq_index=seq_index)
    start = int(row["batch_start_row"])
    used_length = int(row["used_length"])
    end = start + used_length

    if sparse_format:
        representation = scipy.sparse.load_npz(representation_fpath)
        seq_repr = representation[start:end, :]
    else:
        representation = torch.load(representation_fpath)
        seq_repr = representation[start:end, :]
    return seq_repr, row


def get_sequence_representation_by_layer(
        seq_index: int,
        layer: int,
        csv_fpath: str,
        base_dir: str,
        sparse_format: bool = False,
):
    """
    Resolve the right batch file from CSV and return representation slice for one sequence.

    - embeddings: set sparse_format=False and base_dir=embeddings_dir
    - latents: set sparse_format=True and base_dir=latents_dir
    """
    row = get_sequence_metadata_row(csv_fpath=csv_fpath, seq_index=seq_index)
    batch_index = int(row["batch_index"])

    ext = "npz" if sparse_format else "pt"
    representation_fpath = f"{base_dir}batch_{batch_index:06d}-layer_{layer}.{ext}"
    seq_repr, row = parse_concatenated_representation(
        representation_fpath=representation_fpath,
        csv_fpath=csv_fpath,
        seq_index=seq_index,
        sparse_format=sparse_format,
    )
    return seq_repr, row, representation_fpath
