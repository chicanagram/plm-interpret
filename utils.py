import torch
import scipy
import re
from pathlib import Path
import time

_WINDOWS_BAD = r'[<>:"/\\|?*]'
_CONTROL_CHARS = r"[\x00-\x1f\x7f]"  # includes \n \r \t etc.

def safe_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(_CONTROL_CHARS, "_", name)
    name = re.sub(_WINDOWS_BAD, "_", name)
    name = name.strip(" .")  # Windows forbids trailing space/dot
    if not name:
        name = "seq"
    return name[:max_len]


def safe_torch_save(tensor, path: Path, retries=3):
    for attempt in range(retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path)
            return
        except RuntimeError as e:
            if attempt == retries - 1:
                raise
            print(f"Save failed, retrying ({attempt+1}/{retries})...")
            time.sleep(1)


def safe_numpy_save(array, path: Path, retries=3):
    for attempt in range(retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            scipy.sparse.save_npz(path, array)
            return
        except RuntimeError as e:
            if attempt == retries - 1:
                raise
            print(f"Save failed, retrying ({attempt+1}/{retries})...")
            time.sleep(1)


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