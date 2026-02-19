import csv
import os
import stat
import time
from pathlib import Path
from typing import Sequence, Dict


def chunked(iterable: Sequence, chunk_size: int):
    for start in range(0, len(iterable), chunk_size):
        yield start, iterable[start:start + chunk_size]


def initialize_batch_metadata_csv(
        sequences: Sequence[str],
        seq_names: Sequence[str],
        seq_batch_size: int,
        max_length: int,
        csv_fpath: str
):
    """
    Create metadata CSV with one row per sequence and per-batch row offsets.
    """
    Path(csv_fpath).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_index",
        "seq_name",
        "batch_index",
        "sequence_length",
        "used_length",
        "batch_start_row",
        "embedding_obtained",
        "latent_obtained",
    ]
    with open(csv_fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch_start, batch_seqs in chunked(sequences, seq_batch_size):
            batch_index = batch_start // seq_batch_size
            batch_row_start = 0
            for i, seq in enumerate(batch_seqs):
                seq_index = batch_start + i
                used_length = min(len(seq), max_length)
                writer.writerow({
                    "seq_index": seq_index,
                    "seq_name": seq_names[seq_index],
                    "batch_index": batch_index,
                    "sequence_length": len(seq),
                    "used_length": used_length,
                    "batch_start_row": batch_row_start,
                    "embedding_obtained": 0,
                    "latent_obtained": 0,
                })
                batch_row_start += used_length


def update_batch_metadata_flags(
        csv_fpath: str,
        processed_seq_indices,
        embedding_obtained: bool = None,
        latent_obtained: bool = None,
):
    """
    Update embedding/latent obtained flags in CSV for a specific set of processed sequences.
    """
    processed_seq_indices = set(int(i) for i in processed_seq_indices)
    with open(csv_fpath, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        if int(row["seq_index"]) not in processed_seq_indices:
            continue
        if embedding_obtained is not None:
            row["embedding_obtained"] = int(embedding_obtained)
        if latent_obtained is not None:
            row["latent_obtained"] = int(latent_obtained)

    _write_csv_rows_atomic_with_retries(
        csv_fpath=csv_fpath,
        fieldnames=fieldnames,
        rows=rows,
    )


def get_sequence_metadata_row(csv_fpath: str, seq_index: int) -> Dict[str, str]:
    with open(csv_fpath, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next((r for r in reader if int(r["seq_index"]) == seq_index), None)
    if row is None:
        raise ValueError(f"Sequence index {seq_index} not found in metadata CSV: {csv_fpath}")
    return row


def get_batch_progress(csv_fpath: str, processed_seq_indices) -> Dict[str, bool]:
    """
    Return whether all provided sequence indices have embedding/latent flags set.
    """
    processed_seq_indices = set(int(i) for i in processed_seq_indices)
    if not processed_seq_indices:
        return {"all_embedding_obtained": True, "all_latent_obtained": True}

    embedding_flags = []
    latent_flags = []
    with open(csv_fpath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["seq_index"]) in processed_seq_indices:
                embedding_flags.append(int(row["embedding_obtained"]) == 1)
                latent_flags.append(int(row["latent_obtained"]) == 1)

    if len(embedding_flags) != len(processed_seq_indices):
        raise ValueError("Some processed sequence indices were not found in metadata CSV.")

    return {
        "all_embedding_obtained": all(embedding_flags),
        "all_latent_obtained": all(latent_flags),
    }


def _write_csv_rows_atomic_with_retries(
        csv_fpath: str,
        fieldnames,
        rows,
        retries: int = 5,
        sleep_seconds: float = 1.0,
):
    """
    Write CSV via temp file + atomic replace.
    Retries permission errors (e.g. transient lock), and attempts to add user write bit.
    """
    csv_path = Path(csv_fpath)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")

    for attempt in range(retries):
        try:
            with open(tmp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_path, csv_path)
            return
        except PermissionError as exc:
            try:
                current_mode = csv_path.stat().st_mode
                csv_path.chmod(current_mode | stat.S_IWUSR)
            except Exception:
                pass
            if attempt == retries - 1:
                raise PermissionError(
                    f"Could not update metadata CSV due to permission denial: {csv_fpath}. "
                    f"Close any editor/spreadsheet using it and ensure write permission."
                ) from exc
            time.sleep(sleep_seconds)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass


def validate_metadata_csv(
        sequences: Sequence[str],
        seq_names: Sequence[str],
        seq_batch_size: int,
        max_length: int,
        csv_fpath: str,
):
    """
    Validate existing metadata CSV matches the current FASTA order/content and batching params.
    Raises ValueError on mismatch.
    """
    required_fields = [
        "seq_index",
        "seq_name",
        "batch_index",
        "sequence_length",
        "used_length",
        "batch_start_row",
        "embedding_obtained",
        "latent_obtained",
    ]
    with open(csv_fpath, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {csv_fpath}")
        missing_fields = [c for c in required_fields if c not in reader.fieldnames]
        if missing_fields:
            raise ValueError(f"Metadata CSV missing required columns {missing_fields}: {csv_fpath}")
        rows = list(reader)

    if len(rows) != len(sequences):
        raise ValueError(
            f"Metadata CSV row count mismatch for resume: csv_rows={len(rows)} fasta_sequences={len(sequences)}"
        )

    for batch_start, batch_seqs in chunked(sequences, seq_batch_size):
        batch_index = batch_start // seq_batch_size
        expected_start_row = 0
        for i, seq in enumerate(batch_seqs):
            seq_index = batch_start + i
            row = rows[seq_index]

            if int(row["seq_index"]) != seq_index:
                raise ValueError(
                    f"Metadata CSV seq_index mismatch at row {seq_index}: "
                    f"csv={row['seq_index']} expected={seq_index}"
                )
            if row["seq_name"] != seq_names[seq_index]:
                raise ValueError(
                    f"Metadata CSV seq_name mismatch at seq_index={seq_index}: "
                    f"csv={row['seq_name']} expected={seq_names[seq_index]}"
                )
            if int(row["batch_index"]) != batch_index:
                raise ValueError(
                    f"Metadata CSV batch_index mismatch at seq_index={seq_index}: "
                    f"csv={row['batch_index']} expected={batch_index}"
                )
            if int(row["sequence_length"]) != len(seq):
                raise ValueError(
                    f"Metadata CSV sequence_length mismatch at seq_index={seq_index}: "
                    f"csv={row['sequence_length']} expected={len(seq)}"
                )

            expected_used_length = min(len(seq), max_length)
            if int(row["used_length"]) != expected_used_length:
                raise ValueError(
                    f"Metadata CSV used_length mismatch at seq_index={seq_index}: "
                    f"csv={row['used_length']} expected={expected_used_length}"
                )
            if int(row["batch_start_row"]) != expected_start_row:
                raise ValueError(
                    f"Metadata CSV batch_start_row mismatch at seq_index={seq_index}: "
                    f"csv={row['batch_start_row']} expected={expected_start_row}"
                )
            expected_start_row += expected_used_length
