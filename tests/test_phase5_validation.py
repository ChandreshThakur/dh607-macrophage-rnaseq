from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.stats import false_discovery_control

from biological_analysis.phase5_validation import calibrate_interaction_null, reliable_mask


ROOT = Path(__file__).resolve().parents[1]
requires_phase5_outputs = pytest.mark.skipif(
    not (ROOT / "results/tables/phase5_validation_summary.json").is_file(),
    reason="Run scripts/run_phase5.py with the published inputs first",
)


def test_reliable_mask_requires_all_validation_conditions():
    frame = pd.DataFrame({
        "count_reliable": [True, False, True, True, True],
        "same_direction": [True, True, False, True, True],
        "course_adjusted_p_value": [.01, .01, .01, .2, .01],
        "count_adjusted_p_value": [.01, .01, .01, .01, .01],
        "course_interaction_log2": [.8, .8, .8, .8, .2],
    })
    assert reliable_mask(frame).tolist() == [True, False, False, False, False]


def test_centered_null_is_reproducible_and_near_nominal():
    rng = np.random.default_rng(11)
    contrasts = pd.DataFrame(rng.normal(size=(500, 40)))
    first, summary = calibrate_interaction_null(contrasts, n_permutations=80, seed=9)
    second, _ = calibrate_interaction_null(contrasts, n_permutations=80, seed=9)
    pd.testing.assert_frame_equal(first, second)
    assert 0.03 < summary["mean_nominal_p_lt_0_05_fraction"] < 0.07


@requires_phase5_outputs
def test_phase5_required_outputs_and_ifnl_invariants():
    required = [
        "results/tables/phase5_gene_reliability.csv",
        "results/tables/ifnl_gene_validation.csv",
        "results/tables/negative_interaction_interpretation.csv",
        "results/tables/phase5_enrichment_results.csv",
        "results/figures/ifnl_four_condition_expression.png",
        "results/figures/ifnl_donor_response.png",
        "results/figures/positive_negative_interaction_comparison.png",
        "results/figures/interaction_vs_expression_abundance.png",
    ]
    for relative in required:
        path = ROOT / relative
        assert path.exists() and path.stat().st_size > 0, relative

    ifnl = pd.read_csv(ROOT / "results/tables/ifnl_gene_validation.csv")
    assert set(ifnl.gene_name) == {"IFNL1", "IFNL2", "IFNL3"}
    assert ifnl.gene_id.is_unique
    assert set(ifnl.chr.astype(str)) == {"19"}
    assert ifnl.count_reliable.all()
    assert (ifnl.positive_donor_fraction == 1).all()
    assert (ifnl.raw_mean_D > ifnl[["raw_mean_A", "raw_mean_B", "raw_mean_C"]].max(axis=1)).all()
    for column in [c for c in ifnl if c.startswith("interaction_log2_pseudocount_")]:
        assert (ifnl[column] > 0).all()


@requires_phase5_outputs
def test_negative_panel_and_enrichment_integrity():
    negative = pd.read_csv(ROOT / "results/tables/negative_interaction_interpretation.csv")
    assert len(negative) == 11
    assert (negative.interaction_log2 < 0).all()
    assert (negative.mean_D >= negative[["mean_B", "mean_C"]].max(axis=1)).all()
    assert set(negative.interpretation) == {
        "sub-additive without combined-condition suppression"
    }

    enrichment = pd.read_csv(ROOT / "results/tables/phase5_enrichment_results.csv")
    assert not enrichment.empty
    assert set(enrichment.direction) == {"positive", "negative"}
    assert enrichment.p_value.between(0, 1).all()
    assert enrichment.adjusted_p_value.between(0, 1).all()
    assert enrichment.groupby("direction").go_id.nunique().nunique() == 1
    assert enrichment.groupby("go_id").direction.nunique().eq(2).all()
    for _, results in enrichment.groupby("direction"):
        assert (results.overlap_count == 0).any()
        np.testing.assert_allclose(results.adjusted_p_value, false_discovery_control(results.p_value))
    assert (enrichment.selected_genes_annotated <= enrichment.universe_genes_annotated).all()
