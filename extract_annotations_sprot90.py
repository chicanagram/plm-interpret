#!/usr/bin/env python
"""Process sprot90 filtered annotations into InterPLM concept shard format."""

import argparse
import gc
import logging
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


# Allow running from repo root while importing InterPLM package modules.
REPO_ROOT = Path(__file__).resolve().parent
INTERPLM_ROOT = REPO_ROOT / "InterPLM"
if str(INTERPLM_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERPLM_ROOT))

from interplm.analysis.concepts.concept_constants import (  # noqa: E402
    aa_map,
    binary_meta_cols,
    categorical_concepts,
    paired_binary_cols,
)
from interplm.analysis.concepts.parsing_utils import (  # noqa: E402
    analyze_categorical_features,
    process_binary_feature,
    process_categorical_feature,
    process_interaction_feature,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_annotations_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV annotation table from local disk."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input format for {path}")


def add_sequence_features(row: pd.Series) -> pd.Series:
    """Add amino acid identity and local index list for one protein row."""
    split_sequence = list(row["Sequence"])
    row["amino_acid"] = split_sequence
    row["local_index"] = list(range(len(split_sequence)))
    return row


def one_hot_encode(
    df: pd.DataFrame, column: str, mapping: Dict[str, str], include_other: bool = True
) -> pd.DataFrame:
    """One-hot encode a categorical column."""
    categories = list(mapping.keys())
    if include_other:
        categories.append("Other")

    encoding_series = pd.Categorical(
        df[column].apply(lambda x: x if x in mapping else "Other"),
        categories=categories,
    )
    return pd.concat([df, pd.get_dummies(encoding_series, prefix=column)], axis=1)


def preprocess_proteins(
    df: pd.DataFrame,
    min_protein_length: int,
    require_alphafolddb: bool,
    dedupe_by_sequence: bool,
) -> pd.DataFrame:
    """Filter/clean proteins while tolerating filtered custom CSV schemas."""
    required_cols = ["Entry", "Sequence"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    if "Length" not in df.columns:
        df["Length"] = df["Sequence"].astype(str).str.len()
    else:
        df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
        missing_len = df["Length"].isna()
        if missing_len.any():
            df.loc[missing_len, "Length"] = (
                df.loc[missing_len, "Sequence"].astype(str).str.len()
            )
    # # shuffle
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df[df["Length"] <= min_protein_length]

    if require_alphafolddb:
        if "AlphaFoldDB" not in df.columns:
            raise ValueError(
                "require_alphafolddb=True but column 'AlphaFoldDB' is missing."
            )
        df = df[df["AlphaFoldDB"].notnull()]

    if dedupe_by_sequence:
        df = df.drop_duplicates(subset=["Sequence"], keep="first")

    return df.reset_index(drop=True)


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all concept columns expected by InterPLM exist."""
    expected = set(binary_meta_cols + paired_binary_cols)
    expected.update([x[0] for x in categorical_concepts])
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df


def enumerate_protein_subcategories(
    df: pd.DataFrame, min_required_instances: int
) -> Dict[str, List[str]]:
    """Find categorical subtypes with enough support."""
    categorical_options = {}
    for col_name, col_shortname, col_separator in categorical_concepts:
        if col_name not in df.columns or df[col_name].dropna().empty:
            categorical_options[col_name] = ["any"]
            continue
        _, _, _, notes = analyze_categorical_features(
            df, col_name, col_shortname, col_separator
        )
        notes = notes[notes >= min_required_instances]
        categorical_options[col_name] = [c for c in notes.keys() if c != ""] + ["any"]
    return categorical_options


# def shard_protein_data(df: pd.DataFrame, output_dir: Path, n_shards: int) -> None:
#     """Split protein rows into output shard directories."""
#     np.random.seed(42)
#     shards = np.array_split(df, n_shards)
#     for shard_id, shard_df in enumerate(shards):
#         shard_dir = output_dir / f"shard_{shard_id}"
#         shard_dir.mkdir(parents=True, exist_ok=True)
#         shard_df.reset_index(drop=True).to_csv(
#             shard_dir / "protein_data.tsv", sep="\t", index=False
#         )
#         logger.info("Wrote shard %s with %s proteins", shard_id, len(shard_df))


def shard_protein_data(df: pd.DataFrame, output_dir: Path, shard_size: int) -> None:
    """Split protein rows into output shard directories."""
    np.random.seed(42)
    n_shards = np.ceil(len(df) / shard_size).astype(int)
    print(f'Generating {n_shards} shards...')
    for shard_id in range(n_shards):
        i_start = shard_id*shard_size
        i_end = min((shard_id+1)*shard_size, len(df))
        shard_df = df.iloc[i_start:i_end].copy()
        shard_dir = output_dir / f"shard_{shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_df.reset_index(drop=True).to_csv(
            shard_dir / "protein_data.tsv", sep="\t", index=False
        )
        logger.info("Wrote shard %s with %s proteins", shard_id, len(shard_df))


def expand_features(
    df: pd.DataFrame,
    categorical_column_options: Dict[str, List[str]],
    binary_cols: List[str],
    interaction_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Expand protein-level annotations to residue-level vectors."""
    new_columns = defaultdict(list)

    for col, category_options in categorical_column_options.items():
        current_index = {cat: 1 for cat in category_options}
        logger.info("Processing categorical column: %s", col)

        if col not in df.columns or df[col].isnull().all():
            for category_option in category_options:
                new_columns[f"{col}_{category_option}"] = df["Length"].apply(
                    lambda x: [False] * int(x)
                )
            continue

        col_name = df[col].dropna().iloc[0].split(" ")[0]
        for _, row in df.iterrows():
            results, current_index = process_categorical_feature(
                row[col], col_name, category_options, int(row["Length"]), current_index
            )
            for category_option, result in zip(category_options, results):
                new_columns[f"{col}_{category_option}"].append(result)

    for col in binary_cols:
        current_index = 1
        logger.info("Processing binary column: %s", col)
        if col not in df.columns or df[col].isnull().all():
            new_columns[f"{col}_binary"] = df["Length"].apply(
                lambda x: [False] * int(x)
            )
            continue

        col_name = df[col].dropna().iloc[0].split(" ")[0]
        for _, row in df.iterrows():
            result, current_index = process_binary_feature(
                row[col], col_name, int(row["Length"]), current_index
            )
            new_columns[f"{col}_binary"].append(result)

    for col in interaction_cols:
        logger.info("Processing interaction column: %s", col)
        if col not in df.columns or df[col].isnull().all():
            new_columns[f"{col}_binary"] = df["Length"].apply(
                lambda x: [False] * int(x)
            )
            continue

        col_name = df[col].dropna().iloc[0].split(" ")[0]
        for _, row in df.iterrows():
            indices, _ = process_interaction_feature(
                row[col], col_name, int(row["Length"])
            )
            new_columns[f"{col}_binary"].append(indices)

    return pd.concat([df, pd.DataFrame(new_columns)], axis=1), list(new_columns.keys())


def convert_shard_to_amino_acid_features(
    shard_id: int,
    input_path: Path,
    output_dir: Path,
    categorical_options: Dict[str, List[str]],
    binary_cols: List[str],
    interaction_cols: List[str],
    overwrite: bool,
    concept_columns_filename: str,
) -> None:
    """Convert one protein shard to amino-acid-level concept sparse matrix."""
    output_metadata = output_dir / f"shard_{shard_id}" / "aa_metadata.csv"
    output_sparse = output_dir / f"shard_{shard_id}" / "aa_concepts.npz"
    if output_sparse.exists() and not overwrite:
        logger.info("Shard %s already processed, skipping", shard_id)
        return

    logger.info("Converting shard %s", shard_id)
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    df = ensure_feature_columns(df)
    df = df.apply(add_sequence_features, axis=1)

    df, new_cols = expand_features(
        df=df,
        categorical_column_options=categorical_options,
        binary_cols=binary_cols,
        interaction_cols=interaction_cols,
    )

    cols_to_expand = ["amino_acid", "local_index"] + new_cols
    df = df[["Entry"] + cols_to_expand].explode(cols_to_expand).reset_index(drop=True)
    df.columns = [re.sub(r"_binary", "", col) for col in df.columns]
    df.columns = [re.sub(r" \[FT\]", "", col) for col in df.columns]
    df = one_hot_encode(df, "amino_acid", aa_map, include_other=True)

    metadata = df.iloc[:, :3]
    metadata.to_csv(output_metadata, index=False)

    concept_cols_path = output_dir / concept_columns_filename
    if not concept_cols_path.exists():
        concept_cols_path.write_text("\n".join(df.columns[3:]))

    feature_matrix = sparse.csr_matrix(df.iloc[:, 3:].astype(np.uint32))
    sparse.save_npz(output_sparse, feature_matrix)
    logger.info("Shard %s complete (%s amino acids)", shard_id, len(df))


def run(
    input_annotations_path: Path,
    output_dir: Path,
    shard_size: int = 1000,
    min_required_instances: int = 10,
    min_protein_length: int = 1536,
    require_alphafolddb: bool = False,
    dedupe_by_sequence: bool = True,
    overwrite: bool = False,
    max_workers: int = 1,
    concept_columns_filename: str = "uniprotkb_aa_concepts_columns.txt",
) -> None:
    """Execute end-to-end conversion from filtered annotation CSV -> concept shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading annotation table from %s", input_annotations_path)
    df = read_annotations_table(input_annotations_path)
    logger.info("Loaded %s rows, %s columns", len(df), len(df.columns))

    df = ensure_feature_columns(df)
    df = preprocess_proteins(
        df=df,
        min_protein_length=min_protein_length,
        require_alphafolddb=require_alphafolddb,
        dedupe_by_sequence=dedupe_by_sequence,
    )
    logger.info("After filtering: %s proteins", len(df))

    # shard_protein_data(df=df, output_dir=output_dir, n_shards=n_shards)
    shard_protein_data(df=df, output_dir=output_dir, shard_size=shard_size)
    n_shards = np.ceil(len(df)/shard_size).astype(int)
    categorical_options = enumerate_protein_subcategories(
        df=df, min_required_instances=min_required_instances
    )

    if max_workers is None or max_workers < 1:
        max_workers = 1

    logger.info("Using max_workers=%s for shard conversion", max_workers)
    if max_workers == 1:
        # Avoid ProcessPool when running single-worker mode to reduce memory overhead
        # and prevent BrokenProcessPool from child-process termination.
        for shard_id in range(n_shards):
            try:
                convert_shard_to_amino_acid_features(
                    shard_id=shard_id,
                    input_path=output_dir / f"shard_{shard_id}" / "protein_data.tsv",
                    output_dir=output_dir,
                    categorical_options=categorical_options,
                    binary_cols=binary_meta_cols,
                    interaction_cols=paired_binary_cols,
                    overwrite=overwrite,
                    concept_columns_filename=concept_columns_filename,
                )
                gc.collect()
            except Exception as exc:
                logger.error("Shard %s failed: %s", shard_id, exc)
                raise
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    convert_shard_to_amino_acid_features,
                    shard_id=shard_id,
                    input_path=output_dir / f"shard_{shard_id}" / "protein_data.tsv",
                    output_dir=output_dir,
                    categorical_options=categorical_options,
                    binary_cols=binary_meta_cols,
                    interaction_cols=paired_binary_cols,
                    overwrite=overwrite,
                    concept_columns_filename=concept_columns_filename,
                ): shard_id
                for shard_id in range(n_shards)
            }
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Shard %s failed: %s", shard_id, exc)
                    raise

    logger.info("All shards processed into %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert sprot90 filtered annotation table to InterPLM concept shards."
    )
    parser.add_argument("--input_annotations_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    # parser.add_argument("--n_shards", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--min_required_instances", type=int, default=10)
    parser.add_argument("--min_protein_length", type=int, default=1536)
    parser.add_argument("--require_alphafolddb", action="store_true")
    parser.add_argument("--no_dedupe_by_sequence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of shard workers. Use 1 for low-memory safe mode (default).",
    )
    parser.add_argument(
        "--concept_columns_filename",
        type=str,
        default="uniprotkb_aa_concepts_columns.txt",
    )
    return parser


if __name__ == "__main__":
    # IDE mode: set True to run with hardcoded values below.
    # CLI mode: keep False to use command-line args.
    USE_HARDCODED_CONFIG = True

    if USE_HARDCODED_CONFIG:
        hardcoded_config = {
            "input_annotations_path": Path(
                "/Users/charmainechia/Documents/projects/seq-db/uniprot_sprot/raw/sprot90_filtered_annotations_ordered.csv"
            ),
            "output_dir": Path(
                "/Users/charmainechia/Documents/projects/seq-db/uniprot_sprot/processed_sprot90"
            ),
            # "n_shards": 256,
            "shard_size": 1000,
            "min_required_instances": 100,
            "min_protein_length": 50000,
            "require_alphafolddb": False,
            "dedupe_by_sequence": False,
            "overwrite": False,
            "max_workers": 1,
            "concept_columns_filename": "uniprotkb_aa_concepts_columns.txt",
        }

        # df = pd.read_csv(hardcoded_config['input_annotations_path'])
        # print(df.columns)
        # df = df.drop(columns=['Unnamed: 0'])
        # df = df.sort_values(by='seq_idx_in_fasta')
        # print(df[['Entry','Entry Name']].head())
        # df.to_csv(str(hardcoded_config['input_annotations_path']).replace('.csv','_ordered.csv'))

        run(**hardcoded_config)
    else:
        args = build_parser().parse_args()
        run(
            input_annotations_path=args.input_annotations_path,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            min_required_instances=args.min_required_instances,
            min_protein_length=args.min_protein_length,
            require_alphafolddb=args.require_alphafolddb,
            dedupe_by_sequence=not args.no_dedupe_by_sequence,
            overwrite=args.overwrite,
            max_workers=args.max_workers,
            concept_columns_filename=args.concept_columns_filename,
        )
