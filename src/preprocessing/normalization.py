"""Transparent normalization utilities used by the course-level analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def median_of_ratios(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """DESeq-style median-of-ratios normalization using all-positive genes.

    Genes containing a zero in any sample are excluded only from size-factor
    estimation, not from the returned normalized matrix.
    """
    if counts.empty or (counts.to_numpy() < 0).any():
        raise ValueError("counts must be a nonempty, nonnegative matrix")
    values = counts.to_numpy(dtype=float)
    eligible = (values > 0).all(axis=1)
    if not eligible.any():
        raise ValueError("no genes are positive in every sample")
    geometric_means = np.exp(np.log(values[eligible]).mean(axis=1))
    ratios = values[eligible] / geometric_means[:, None]
    factors = np.median(ratios, axis=0)
    if (factors <= 0).any() or not np.isfinite(factors).all():
        raise ValueError("invalid size factors")
    factors = factors / np.exp(np.log(factors).mean())
    factor_series = pd.Series(factors, index=counts.columns, name="size_factor")
    normalized = counts.div(factor_series, axis=1)
    return normalized, factor_series


def expressed_filter(counts: pd.DataFrame, minimum_count: int = 10, minimum_samples: int = 84) -> pd.Series:
    """Return a documented prevalence-based low-expression filter."""
    if minimum_count < 0 or minimum_samples < 1:
        raise ValueError("filter thresholds must be positive")
    return (counts >= minimum_count).sum(axis=1) >= minimum_samples


def log2_transform(normalized_counts: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    """Log2 pseudocount transform; deliberately not called a VST or Anscombe transform."""
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    return np.log2(normalized_counts + pseudocount)

