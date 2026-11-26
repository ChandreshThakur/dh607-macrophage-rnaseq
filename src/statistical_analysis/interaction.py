"""Within-donor factorial contrasts and count-model design validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = ("A", "B", "C", "D")
CONDITION_CODING = {
    "A": (0, 0),
    "B": (1, 0),
    "C": (0, 1),
    "D": (1, 1),
}


def benjamini_hochberg(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or np.isnan(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    n = len(values)
    order = np.argsort(values)
    ranked = values[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def interaction_contrasts(expression: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute D - B - C + A for each gene and donor."""
    needed = {"sample_id", "donor", "condition"}
    if missing := needed.difference(metadata.columns):
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    if metadata.duplicated(["donor", "condition"]).any():
        raise ValueError("each donor-condition combination must be unique")
    lookup = metadata.set_index(["donor", "condition"])["sample_id"]
    donors = sorted(metadata["donor"].unique())
    missing_pairs = [(d, c) for d in donors for c in CONDITIONS if (d, c) not in lookup.index]
    if missing_pairs:
        raise ValueError(f"incomplete donor-condition design: {missing_pairs[:5]}")
    sample_ids = set(metadata["sample_id"])
    if not sample_ids.issubset(expression.columns):
        raise ValueError("expression matrix is missing metadata samples")
    arrays = {
        condition: expression.loc[:, [lookup.loc[(donor, condition)] for donor in donors]].to_numpy()
        for condition in CONDITIONS
    }
    values = arrays["D"] - arrays["B"] - arrays["C"] + arrays["A"]
    return pd.DataFrame(values, index=expression.index, columns=donors)


def validate_factorial_metadata(
    metadata: pd.DataFrame, sample_ids: pd.Index | list[str] | None = None
) -> pd.DataFrame:
    """Validate the paired 2x2 design and return explicit binary factors."""
    needed = {"sample_id", "donor", "condition", "condition_name", "IFNg", "SL1344"}
    if missing := needed.difference(metadata.columns):
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    if metadata.sample_id.duplicated().any():
        raise ValueError("sample identifiers must be unique")
    if metadata.duplicated(["donor", "condition"]).any():
        raise ValueError("donor-condition combinations must be unique")
    observed = set(metadata.condition)
    if observed != set(CONDITIONS):
        raise ValueError(f"conditions must be exactly A/B/C/D, observed {sorted(observed)}")
    counts = metadata.groupby(["donor", "condition"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CONDITIONS, fill_value=0)
    if not (counts == 1).all().all():
        raise ValueError("every donor must have exactly one A/B/C/D observation")
    observed_mapping = (
        metadata.groupby("condition", observed=True)[["condition_name", "IFNg", "SL1344"]]
        .agg(lambda x: tuple(pd.unique(x)))
    )
    expected = {
        "A": (("naive",), ("naive",), ("control",)),
        "B": (("IFNg",), ("primed",), ("control",)),
        "C": (("SL1344",), ("naive",), ("infected",)),
        "D": (("IFNg_SL1344",), ("primed",), ("infected",)),
    }
    for condition, values in expected.items():
        actual = tuple(observed_mapping.loc[condition, col] for col in ("condition_name", "IFNg", "SL1344"))
        if actual != values:
            raise ValueError(f"unexpected metadata coding for condition {condition}: {actual}")
    if sample_ids is not None:
        supplied = pd.Index(sample_ids)
        if supplied.has_duplicates or set(supplied) != set(metadata.sample_id):
            raise ValueError("expression samples and metadata sample IDs must match exactly")
    validated = metadata.copy()
    validated["IFNg_binary"] = validated.condition.map({k: v[0] for k, v in CONDITION_CODING.items()}).astype(int)
    validated["Salmonella_binary"] = validated.condition.map({k: v[1] for k, v in CONDITION_CODING.items()}).astype(int)
    return validated


def fixed_effect_design_matrix(metadata: pd.DataFrame) -> pd.DataFrame:
    """Build the documented donor-blocked factorial matrix without a formula parser."""
    validated = validate_factorial_metadata(metadata)
    donors = pd.get_dummies(validated.donor.astype(str), prefix="donor", drop_first=True, dtype=float)
    design = pd.DataFrame({"Intercept": np.ones(len(validated))}, index=validated.index)
    design = pd.concat([design, donors.set_axis(validated.index)], axis=1)
    design["IFNg_binary"] = validated.IFNg_binary.to_numpy(dtype=float)
    design["Salmonella_binary"] = validated.Salmonella_binary.to_numpy(dtype=float)
    design["IFNg_binary:Salmonella_binary"] = (
        validated.IFNg_binary * validated.Salmonella_binary
    ).to_numpy(dtype=float)
    return design


def validate_design_matrix(design: pd.DataFrame) -> dict[str, int | float | bool]:
    """Return estimability diagnostics and fail if the design is rank deficient."""
    values = design.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(values))
    full_rank = rank == design.shape[1]
    diagnostics = {
        "n_rows": int(design.shape[0]),
        "n_columns": int(design.shape[1]),
        "rank": rank,
        "full_rank": full_rank,
        "condition_number": float(np.linalg.cond(values)),
    }
    if not full_rank:
        raise ValueError(f"design matrix is not full rank: {rank}/{design.shape[1]}")
    return diagnostics


def interaction_coefficient_contrast(columns: pd.Index | list[str]) -> np.ndarray:
    """Return a verified unit contrast for the exact factorial interaction column."""
    names = list(columns)
    target = "IFNg_binary:Salmonella_binary"
    if names.count(target) != 1:
        raise ValueError(f"expected exactly one {target!r} coefficient, found {names.count(target)}")
    contrast = np.zeros(len(names), dtype=float)
    contrast[names.index(target)] = 1.0
    return contrast


def interaction_influence(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize donor sign consistency and exact leave-one-donor-out influence."""
    values = contrasts.to_numpy(dtype=float)
    n = values.shape[1]
    if n < 3:
        raise ValueError("at least three donors are required for influence diagnostics")
    means = values.mean(axis=1)
    loo = (values.sum(axis=1, keepdims=True) - values) / (n - 1)
    max_shift = np.max(np.abs(loo - means[:, None]), axis=1)
    max_abs = np.max(np.abs(values), axis=1)
    positive_fraction = (values > 0).mean(axis=1)
    negative_fraction = (values < 0).mean(axis=1)
    return pd.DataFrame(
        {
            "median_donor_interaction": np.median(values, axis=1),
            "positive_donor_fraction": positive_fraction,
            "negative_donor_fraction": negative_fraction,
            "majority_direction_fraction": np.maximum(positive_fraction, negative_fraction),
            "maximum_absolute_donor_interaction": max_abs,
            "maximum_leave_one_out_mean_shift": max_shift,
        },
        index=contrasts.index,
    )


def test_interactions(contrasts: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Two-sided one-sample t-tests of independent donor contrasts against zero."""
    values = contrasts.to_numpy(dtype=float)
    n = values.shape[1]
    if n < 2:
        raise ValueError("at least two donors are required")
    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1, ddof=1)
    standard_errors = standard_deviations / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = means / standard_errors
    statistics = np.where((standard_errors == 0) & (means == 0), 0.0, statistics)
    pvalues = 2 * stats.t.sf(np.abs(statistics), df=n - 1)
    pvalues = np.where((standard_errors == 0) & (means == 0), 1.0, pvalues)
    critical = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    result = pd.DataFrame(
        {
            "mean_interaction_log2": means,
            "standard_deviation": standard_deviations,
            "standard_error": standard_errors,
            "ci_lower": means - critical * standard_errors,
            "ci_upper": means + critical * standard_errors,
            "t_statistic": statistics,
            "degrees_of_freedom": n - 1,
            "p_value": pvalues,
        },
        index=contrasts.index,
    )
    result["adjusted_p_value"] = benjamini_hochberg(result["p_value"].to_numpy())
    return result.sort_values(["adjusted_p_value", "p_value"])
