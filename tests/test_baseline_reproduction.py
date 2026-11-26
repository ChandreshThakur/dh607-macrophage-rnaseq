import numpy as np
import pandas as pd
import pytest

from biological_analysis.baseline_reproduction import effect_concordance, paired_effect_table


def test_paired_effect_table_uses_within_donor_differences():
    metadata = pd.DataFrame({
        "sample_id": ["d1_A", "d1_B", "d2_A", "d2_B"],
        "donor": ["d1", "d1", "d2", "d2"],
        "condition": ["A", "B", "A", "B"],
    })
    expression = pd.DataFrame(
        [[1.0, 3.0, 10.0, 14.0], [5.0, 5.0, 8.0, 8.0]],
        index=["g1", "g2"], columns=metadata.sample_id,
    )
    result, contrasts = paired_effect_table(expression, metadata, "B")
    assert contrasts.loc["g1"].tolist() == [2.0, 4.0]
    assert result.loc["g1", "mean_paired_log2_change"] == 3.0
    assert result.loc["g2", "p_value"] == 1.0


def test_effect_concordance_threshold_and_alignment():
    author = pd.Series([1.0, -1.0, 0.1], index=["g1", "g2", "g3"])
    course = pd.Series([2.0, -2.0, -0.1], index=["g1", "g2", "g3"])
    result = effect_concordance(author, course, threshold=0.5)
    assert result["n_genes_above_threshold_both"] == 2
    assert result["same_direction_fraction_above_threshold"] == 1.0
    with pytest.raises(ValueError, match="identical gene IDs"):
        effect_concordance(author, course.rename(index={"g1": "other"}))
