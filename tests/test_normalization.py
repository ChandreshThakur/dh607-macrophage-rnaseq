import numpy as np
import pandas as pd

from preprocessing.normalization import expressed_filter, log2_transform, median_of_ratios


def test_median_of_ratios_recovers_library_scaling():
    counts = pd.DataFrame({"s1": [10, 20, 30], "s2": [20, 40, 60]}, index=["g1", "g2", "g3"])
    normalized, factors = median_of_ratios(counts)
    assert np.allclose(normalized.s1, normalized.s2)
    assert np.isclose(np.exp(np.log(factors).mean()), 1)


def test_filter_and_log_transform():
    counts = pd.DataFrame({"a": [10, 0], "b": [11, 10]}, index=["keep", "drop"])
    assert expressed_filter(counts, 10, 2).to_dict() == {"keep": True, "drop": False}
    assert np.isclose(log2_transform(pd.DataFrame([[0.0]]), 0.5).iloc[0, 0], -1)

