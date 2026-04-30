#!/usr/bin/env python
"""Map Swiss-Prot FASTA entries to sharded annotation CSV rows and extract matches."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_uniprot_id_from_fasta_header(header: str) -> str:
    """Parse UniProt accession from FASTA header line (without leading '>')."""
    token = header.split()[0]
    parts = token.split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return token


def load_fasta_index_by_id(fasta_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """Return ordered UniProt IDs from FASTA and mapping from ID -> index."""
    fasta_ids: List[str] = []
    id_to_fasta_idx: Dict[str, int] = {}

    with open(fasta_path, "r") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue

            seq_id = parse_uniprot_id_from_fasta_header(line[1:].strip())
            fasta_idx = len(fasta_ids)
            fasta_ids.append(seq_id)
            if seq_id not in id_to_fasta_idx:
                id_to_fasta_idx[seq_id] = fasta_idx

    return fasta_ids, id_to_fasta_idx


def build_fasta_mapping_records(fasta_ids: List[str]) -> List[Dict[str, Optional[object]]]:
    """Initialize mapping records keyed by FASTA index."""
    return [
        {
            "seq_idx_in_fasta": i,
            "csv_suffix": None,
            "seq_idx_in_csv": None,
            "seq_id": seq_id,
        }
        for i, seq_id in enumerate(fasta_ids)
    ]


def filter_annotations_by_fasta(
    fasta_path: Path,
    annotations_dir: Path,
    mapping_output_csv: Path,
    filtered_annotations_output_csv: Path,
    csv_prefix: str = "swissprot_",
    csv_suffix: str = ".csv",
    csv_upper_exclusive: int = 12,
    entry_column: str = "Entry",
) -> None:
    """
    Build FASTA-to-CSV-row mapping and extract matched annotation rows.

    Args:
        fasta_path: Path to clustered FASTA.
        annotations_dir: Directory containing sharded annotation CSV files.
        mapping_output_csv: Output path for FASTA index -> CSV row mapping.
        filtered_annotations_output_csv: Output path for merged filtered annotations.
        csv_prefix: CSV shard filename prefix (e.g., swissprot_).
        csv_suffix: CSV shard filename suffix (e.g., .csv).
        csv_upper_exclusive: Process shard suffixes in range [0, csv_upper_exclusive).
        entry_column: Column containing Swiss-Prot ID/accession in annotation CSVs.
    """
    print(f"Loading FASTA: {fasta_path}")
    fasta_ids, id_to_fasta_idx = load_fasta_index_by_id(fasta_path)
    print(f"Loaded {len(fasta_ids):,} FASTA entries")

    mapping_records = build_fasta_mapping_records(fasta_ids)
    matched_fasta_indices = set()

    print(
        f"Scanning annotation shards in {annotations_dir} "
        f"for suffixes 0..{csv_upper_exclusive - 1}"
    )
    for shard_idx in range(csv_upper_exclusive):
        csv_path = annotations_dir / f"{csv_prefix}{shard_idx}{csv_suffix}"
        if not csv_path.exists():
            print(f"Skipping missing shard: {csv_path}")
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        if entry_column not in df.columns:
            raise ValueError(f"{csv_path} does not contain '{entry_column}' column.")

        for row_idx, seq_id in enumerate(df[entry_column].astype(str).tolist()):
            fasta_idx = id_to_fasta_idx.get(seq_id)
            if fasta_idx is None:
                continue

            record = mapping_records[fasta_idx]
            if record["csv_suffix"] is None:
                record["csv_suffix"] = shard_idx
                record["seq_idx_in_csv"] = row_idx
                matched_fasta_indices.add(fasta_idx)

        print(
            f"Processed {csv_path.name}: {len(df):,} rows "
            f"(matched FASTA entries so far: {len(matched_fasta_indices):,})"
        )

    mapping_df = pd.DataFrame(mapping_records).sort_values(
        "seq_idx_in_fasta"
    ).reset_index(drop=True)

    mapping_output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(mapping_output_csv, index=False)
    print(f"Saved mapping CSV: {mapping_output_csv}")

    matched_df = mapping_df.dropna(subset=["csv_suffix", "seq_idx_in_csv"]).copy()
    matched_df["csv_suffix"] = matched_df["csv_suffix"].astype(int)
    matched_df["seq_idx_in_csv"] = matched_df["seq_idx_in_csv"].astype(int)
    matched_df = matched_df.sort_values(["csv_suffix", "seq_idx_in_csv"]).reset_index(
        drop=True
    )

    filtered_chunks: List[pd.DataFrame] = []
    for shard_idx, group in matched_df.groupby("csv_suffix", sort=True):
        csv_path = annotations_dir / f"{csv_prefix}{shard_idx}{csv_suffix}"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected shard missing for matched rows: {csv_path}"
            )

        df = pd.read_csv(csv_path, low_memory=False)
        row_indices = group["seq_idx_in_csv"].tolist()
        shard_filtered = df.iloc[row_indices].copy()
        shard_filtered.insert(0, "csv_suffix", shard_idx)
        shard_filtered.insert(1, "seq_idx_in_csv", row_indices)
        shard_filtered.insert(2, "seq_idx_in_fasta", group["seq_idx_in_fasta"].tolist())
        filtered_chunks.append(shard_filtered)

    if filtered_chunks:
        filtered_annotations_df = pd.concat(filtered_chunks, ignore_index=True)
    else:
        filtered_annotations_df = pd.DataFrame()

    filtered_annotations_output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_annotations_df.to_csv(filtered_annotations_output_csv, index=False)
    print(f"Saved filtered annotations CSV: {filtered_annotations_output_csv}")
    print(
        f"Done. FASTA total={len(fasta_ids):,}, matched={len(matched_df):,}, "
        f"unmatched={len(fasta_ids) - len(matched_df):,}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build FASTA-to-annotation mapping and extract filtered rows from "
            "sharded Swiss-Prot annotation CSVs."
        )
    )
    parser.add_argument("--fasta_path", type=Path, required=True)
    parser.add_argument("--annotations_dir", type=Path, required=True)
    parser.add_argument("--mapping_output_csv", type=Path, required=True)
    parser.add_argument("--filtered_annotations_output_csv", type=Path, required=True)
    parser.add_argument("--csv_prefix", type=str, default="swissprot_")
    parser.add_argument("--csv_suffix", type=str, default=".csv")
    parser.add_argument("--csv_upper_exclusive", type=int, default=12)
    parser.add_argument("--entry_column", type=str, default="Entry")
    return parser


if __name__ == "__main__":
    # IDE mode: set True to run with hardcoded values below.
    # CLI mode: keep False to use command-line args.
    USE_HARDCODED_CONFIG = True

    if USE_HARDCODED_CONFIG:
        hardcoded_config = {
            "fasta_path": Path(
                "/Users/charmainechia/Documents/projects/seq-db/uniprot_sprot/raw/uniprot_sprot90.fasta"
            ),
            "annotations_dir": Path(
                "/Users/charmainechia/Documents/projects/seq-db/uniprot_sprot/raw/sequence annotations"
            ),
            "mapping_output_csv": Path(
                "/Users/charmainechia/Documents/projects/seq-db/uniprot_sprot/raw/sprot90_mapping.csv"
            ),
            "filtered_annotations_output_csv": Path(
                "/Users/charmainechia/Documents/projects/seq-db/uniprot_sprot/raw/sprot90_filtered_annotations.csv"
            ),
            "csv_prefix": "swissprot_",
            "csv_suffix": ".csv",
            "csv_upper_exclusive": 12,
            "entry_column": "Entry",
        }
        filter_annotations_by_fasta(**hardcoded_config)
    else:
        args = _build_parser().parse_args()
        filter_annotations_by_fasta(
            fasta_path=args.fasta_path,
            annotations_dir=args.annotations_dir,
            mapping_output_csv=args.mapping_output_csv,
            filtered_annotations_output_csv=args.filtered_annotations_output_csv,
            csv_prefix=args.csv_prefix,
            csv_suffix=args.csv_suffix,
            csv_upper_exclusive=args.csv_upper_exclusive,
            entry_column=args.entry_column,
        )
