import numpy as np
import pandas as pd

from exploratory_analysis.full_cohort_audit import (
    categorical_eta_squared,
    robust_mad_z,
    svd_pca,
)


def test_svd_pca_scores_are_centered_and_variance_ratios_sum_below_one():
    expression = pd.DataFrame(
        {"s1": [1.0, 2.0, 3.0], "s2": [2.0, 3.0, 4.0], "s3": [4.0, 1.0, 2.0]},
        index=["g1", "g2", "g3"],
    )
    scores, variance = svd_pca(expression, n_components=2)
    assert np.allclose(scores.mean(axis=0), 0.0, atol=1e-12)
    assert (variance.explained_variance_ratio >= 0).all()
    assert variance.explained_variance_ratio.sum() <= 1.0 + 1e-12


def test_categorical_eta_squared_detects_perfect_group_separation():
    values = pd.Series([0.0, 0.0, 2.0, 2.0])
    groups = pd.Series(["a", "a", "b", "b"])
    assert categorical_eta_squared(values, groups) == 1.0


def test_robust_mad_z_flags_extreme_value_and_handles_constant_input():
    values = pd.Series([1.0, 1.1, 0.9, 1.0, 20.0])
    assert robust_mad_z(values).iloc[-1] > 3.5
    assert (robust_mad_z(pd.Series([2.0, 2.0])) == 0).all()
