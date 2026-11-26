"""Validation of published sample metadata and expression matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_CONDITIONS = ("A", "B", "C", "D")


def read_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"sample_id", "donor", "condition", "condition_name", "SL1344", "IFNg"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    return frame


def donor_completeness(metadata: pd.DataFrame) -> pd.DataFrame:
    counts = (
        metadata.groupby(["donor", "condition"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=EXPECTED_CONDITIONS, fill_value=0)
        .astype(int)
    )
    counts.columns.name = None
    counts["n_conditions_present"] = (counts[list(EXPECTED_CONDITIONS)] > 0).sum(axis=1)
    counts["one_sample_each"] = (counts[list(EXPECTED_CONDITIONS)] == 1).all(axis=1)
    return counts.reset_index()


def condition_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = ["condition", "condition_name", "IFNg", "SL1344"]
    result = (
        metadata.groupby(columns, dropna=False, observed=True)
        .agg(n_samples=("sample_id", "size"), n_donors=("donor", "nunique"))
        .reset_index()
        .sort_values("condition")
    )
    return result


def read_expression(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def validate_count_matrix(counts: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, object]:
    values = counts.to_numpy()
    return {
        "n_genes": int(counts.shape[0]),
        "n_samples": int(counts.shape[1]),
        "orientation": "genes_by_samples",
        "gene_ids_unique": bool(counts.index.is_unique),
        "sample_ids_unique": bool(counts.columns.is_unique),
        "sample_id_sets_match": set(counts.columns) == set(metadata["sample_id"]),
        "sample_order_matches_metadata": list(counts.columns) == list(metadata["sample_id"]),
        "missing_values": int(pd.isna(values).sum()),
        "negative_values": int((values < 0).sum()),
        "noninteger_values": int((values != np.floor(values)).sum()),
    }


def validate_cqn_matrix(cqn: pd.DataFrame, counts: pd.DataFrame) -> dict[str, object]:
    values = cqn.to_numpy()
    return {
        "n_genes": int(cqn.shape[0]),
        "n_samples": int(cqn.shape[1]),
        "same_gene_ids_and_order_as_counts": list(cqn.index) == list(counts.index),
        "same_sample_ids_and_order_as_counts": list(cqn.columns) == list(counts.columns),
        "missing_values": int(pd.isna(values).sum()),
        "finite_values": bool(np.isfinite(values).all()),
    }

