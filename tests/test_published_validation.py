import pandas as pd

from data_validation.published import condition_summary, donor_completeness, validate_count_matrix


def _metadata():
    rows = []
    names = {"A": "naive", "B": "IFNg", "C": "SL1344", "D": "IFNg_SL1344"}
    for donor in ("d1", "d2"):
        for condition in "ABCD":
            rows.append({
                "sample_id": f"{donor}_{condition}", "donor": donor, "condition": condition,
                "condition_name": names[condition], "IFNg": "primed" if condition in "BD" else "naive",
                "SL1344": "infected" if condition in "CD" else "control",
            })
    return pd.DataFrame(rows)


def test_donor_condition_completeness():
    result = donor_completeness(_metadata())
    assert result["one_sample_each"].all()
    assert (result[["A", "B", "C", "D"]] == 1).all().all()


def test_condition_encoding_and_summary():
    result = condition_summary(_metadata())
    assert result.set_index("condition").loc["D", "condition_name"] == "IFNg_SL1344"
    assert result["n_samples"].tolist() == [2, 2, 2, 2]


def test_count_matrix_orientation_and_matching():
    metadata = _metadata()
    counts = pd.DataFrame([[0] * 8, [1] * 8], index=["g1", "g2"], columns=metadata.sample_id)
    result = validate_count_matrix(counts, metadata)
    assert result["orientation"] == "genes_by_samples"
    assert result["sample_id_sets_match"]
    assert result["missing_values"] == result["negative_values"] == result["noninteger_values"] == 0

