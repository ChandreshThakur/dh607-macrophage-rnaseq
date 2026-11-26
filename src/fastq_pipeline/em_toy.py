"""Small multinomial EM example for teaching ambiguous transcript assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_em(
    class_counts: np.ndarray,
    emission_probabilities: np.ndarray,
    initial_abundance: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit transcript mixture proportions and return iteration history/allocations.

    Rows of ``emission_probabilities`` are fragment classes and columns are
    transcripts. Each transcript column must sum to one.
    """
    counts = np.asarray(class_counts, dtype=float)
    emission = np.asarray(emission_probabilities, dtype=float)
    theta = np.asarray(initial_abundance, dtype=float)
    if counts.ndim != 1 or emission.ndim != 2 or emission.shape[0] != len(counts):
        raise ValueError("fragment classes and emission matrix have incompatible shapes")
    if theta.ndim != 1 or emission.shape[1] != len(theta):
        raise ValueError("initial abundance and transcript dimensions differ")
    if (counts < 0).any() or (emission < 0).any() or (theta <= 0).any():
        raise ValueError("counts/probabilities must be nonnegative and initial abundance positive")
    if not np.allclose(emission.sum(axis=0), 1.0):
        raise ValueError("each transcript's emission probabilities must sum to one")
    theta = theta / theta.sum()
    total = counts.sum()
    if total <= 0:
        raise ValueError("at least one fragment is required")

    def log_likelihood(proportions: np.ndarray) -> float:
        probabilities = emission @ proportions
        if ((probabilities <= 0) & (counts > 0)).any():
            return float("-inf")
        return float(np.sum(counts[counts > 0] * np.log(probabilities[counts > 0])))

    rows = [{"iteration": 0, **{f"theta_{i + 1}": value for i, value in enumerate(theta)}, "log_likelihood": log_likelihood(theta), "max_abs_change": np.nan}]
    allocations = np.zeros_like(emission)
    for iteration in range(1, max_iterations + 1):
        weighted = emission * theta
        denominators = weighted.sum(axis=1)
        if ((denominators <= 0) & (counts > 0)).any():
            raise ValueError("an observed class has zero probability")
        responsibilities = np.divide(weighted, denominators[:, None], out=np.zeros_like(weighted), where=denominators[:, None] > 0)
        allocations = counts[:, None] * responsibilities
        updated = allocations.sum(axis=0) / total
        change = float(np.max(np.abs(updated - theta)))
        theta = updated
        rows.append({"iteration": iteration, **{f"theta_{i + 1}": value for i, value in enumerate(theta)}, "log_likelihood": log_likelihood(theta), "max_abs_change": change})
        if change < tolerance:
            break
    else:
        raise RuntimeError("EM did not converge within max_iterations")
    return pd.DataFrame(rows), allocations

