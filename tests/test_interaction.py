import numpy as np
import pandas as pd
import pytest

from statistical_analysis.interaction import (
    benjamini_hochberg,
    fixed_effect_design_matrix,
    interaction_coefficient_contrast,
    interaction_contrasts,
    interaction_influence,
    test_interactions as run_interaction_tests,
    validate_design_matrix,
    validate_factorial_metadata,
)


def factorial_metadata(donors=("x", "y")):
    details = {
        "A": ("naive", "naive", "control"),
        "B": ("IFNg", "primed", "control"),
        "C": ("SL1344", "naive", "infected"),
        "D": ("IFNg_SL1344", "primed", "infected"),
    }
    return pd.DataFrame([
        {"sample_id": f"{donor}_{condition}", "donor": donor, "condition": condition,
         "condition_name": details[condition][0], "IFNg": details[condition][1],
         "SL1344": details[condition][2]}
        for donor in donors for condition in "ABCD"
    ])


def test_factorial_interaction_contrast():
    metadata = factorial_metadata()
    values = {"x_A": 1, "x_B": 2, "x_C": 3, "x_D": 7, "y_A": 2, "y_B": 3, "y_C": 4, "y_D": 10}
    expression = pd.DataFrame([values], index=["gene"])
    contrast = interaction_contrasts(expression, metadata)
    assert contrast.loc["gene"].to_dict() == {"x": 3, "y": 5}


def test_additive_data_have_zero_interaction_and_sign_is_d_minus_b_minus_c_plus_a():
    metadata = factorial_metadata(("x", "y", "z"))
    additive = {f"{d}_{c}": a + b for d, a in zip(("x", "y", "z"), (1, 2, 3))
                for c, b in zip("ABCD", (0, 2, 4, 6))}
    positive = additive | {f"{d}_D": additive[f"{d}_D"] + 3 for d in ("x", "y", "z")}
    negative = additive | {f"{d}_D": additive[f"{d}_D"] - 4 for d in ("x", "y", "z")}
    expression = pd.DataFrame([additive, positive, negative], index=["additive", "positive", "negative"])
    contrast = interaction_contrasts(expression, metadata)
    assert np.allclose(contrast.loc["additive"], 0)
    assert np.allclose(contrast.loc["positive"], 3)
    assert np.allclose(contrast.loc["negative"], -4)


def test_metadata_alignment_full_rank_design_and_exact_interaction_coefficient():
    metadata = factorial_metadata(("x", "y", "z", "w"))
    validated = validate_factorial_metadata(metadata, metadata.sample_id[::-1])
    assert validated.groupby("condition")[["IFNg_binary", "Salmonella_binary"]].first().to_dict("index") == {
        "A": {"IFNg_binary": 0, "Salmonella_binary": 0},
        "B": {"IFNg_binary": 1, "Salmonella_binary": 0},
        "C": {"IFNg_binary": 0, "Salmonella_binary": 1},
        "D": {"IFNg_binary": 1, "Salmonella_binary": 1},
    }
    design = fixed_effect_design_matrix(metadata)
    diagnostics = validate_design_matrix(design)
    assert diagnostics["full_rank"] and diagnostics["rank"] == 7
    contrast = interaction_coefficient_contrast(design.columns)
    assert contrast.sum() == 1
    assert design.columns[np.flatnonzero(contrast)[0]] == "IFNg_binary:Salmonella_binary"


def test_influence_diagnostics_identify_one_extreme_donor():
    contrasts = pd.DataFrame([[1, 1, 1, 9], [-2, -2, -2, -2]], index=["g1", "g2"])
    influence = interaction_influence(contrasts)
    assert influence.loc["g1", "maximum_leave_one_out_mean_shift"] > 1
    assert influence.loc["g2", "maximum_leave_one_out_mean_shift"] == 0
    assert influence.loc["g2", "negative_donor_fraction"] == 1


def test_duplicate_donor_condition_and_sample_mismatch_are_rejected():
    metadata = factorial_metadata(("x", "y", "z"))
    with pytest.raises(ValueError, match="unique"):
        validate_factorial_metadata(pd.concat([metadata, metadata.iloc[[0]]]))
    with pytest.raises(ValueError, match="must match exactly"):
        validate_factorial_metadata(metadata, metadata.sample_id.iloc[:-1])


def test_pydeseq_formula_exposes_the_verified_interaction_coefficient():
    pytest.importorskip("pydeseq2")
    from pydeseq2.dds import DeseqDataSet

    metadata = validate_factorial_metadata(factorial_metadata(("w", "x", "y", "z")))
    model_metadata = metadata.set_index("sample_id")[["donor", "IFNg_binary", "Salmonella_binary"]]
    counts = pd.DataFrame(100, index=model_metadata.index, columns=["g1", "g2"])
    dds = DeseqDataSet(
        counts=counts,
        metadata=model_metadata,
        design="~ donor + IFNg_binary + Salmonella_binary + IFNg_binary:Salmonella_binary",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
        low_memory=True,
    )
    columns = dds.obsm["design_matrix"].columns
    contrast = interaction_coefficient_contrast(columns)
    assert columns[np.flatnonzero(contrast)[0]] == "IFNg_binary:Salmonella_binary"
    assert np.linalg.matrix_rank(dds.obsm["design_matrix"].to_numpy()) == len(columns)


def test_bh_known_values_and_result_export_columns():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.002]))
    assert np.allclose(adjusted, [0.02, 0.04, 0.04, 0.008])
    contrasts = pd.DataFrame([[1, 2, 3], [0, 0, 0]], index=["g1", "g2"])
    result = run_interaction_tests(contrasts)
    assert {"standard_error", "ci_lower", "ci_upper", "p_value", "adjusted_p_value"}.issubset(result.columns)
    assert result.loc["g2", "p_value"] == 1
