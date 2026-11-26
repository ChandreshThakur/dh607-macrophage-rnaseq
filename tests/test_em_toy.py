import numpy as np
import pytest

from fastq_pipeline.em_toy import run_em


def test_em_converges_and_likelihood_does_not_decrease():
    counts = np.array([30, 60, 10])
    emission = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    history, allocations = run_em(counts, emission, np.array([0.5, 0.5]))
    assert np.allclose(history.iloc[-1][["theta_1", "theta_2"]].to_numpy(dtype=float), [0.75, 0.25], atol=1e-8)
    assert (history.log_likelihood.diff().dropna() >= -1e-10).all()
    assert np.allclose(allocations.sum(axis=1), counts)


def test_em_rejects_non_normalized_emissions():
    with pytest.raises(ValueError, match="sum to one"):
        run_em(np.array([1, 1]), np.ones((2, 2)), np.array([0.5, 0.5]))

