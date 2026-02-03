import torch
import re

_WINDOWS_BAD = r'[<>:"/\\|?*]'

def safe_filename(name: str, max_len: int = 150) -> str:
    name = re.sub(_WINDOWS_BAD, "_", name)
    name = name.strip(" .")  # Windows also hates trailing dot/space
    return name[:max_len] if len(name) > max_len else name

def fetch_sequences_from_fasta(sequence_fpath):
    from Bio import SeqIO
    sequence_names = []
    sequence_list = []
    sequence_descriptions = []
    for j, record in enumerate(SeqIO.parse(sequence_fpath, "fasta")):
        sequence_names.append(record.id)
        sequence_list.append(str(record.seq))
        sequence_descriptions.append(record.description)
    return sequence_list, sequence_names, sequence_descriptions


def get_best_device() -> torch.device:
    # 1) NVIDIA GPU (Windows/Linux, CUDA)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 2) Apple Silicon GPU (macOS, MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 3) Fallback
    return torch.device("cpu")