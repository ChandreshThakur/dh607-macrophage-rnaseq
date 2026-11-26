import json

import numpy as np
import pandas as pd
import pytest

from fastq_pipeline.quantification import (
    aggregate_to_genes,
    direction_agreement,
    log2_cpm,
    read_salmon_quant,
)


def test_aggregate_to_genes_sums_counts_and_tpm():
    quant = pd.DataFrame(
        {
            "NumReads": [2.5, 3.5, 7.0],
            "TPM": [10.0, 15.0, 75.0],
        },
        index=pd.Index(["tx1", "tx2", "tx3"], name="Name"),
    )
    mapping = pd.DataFrame(
        {"transcript_id": ["tx1", "tx2", "tx3"], "gene_id": ["g1", "g1", "g2"]}
    )
    counts, tpm = aggregate_to_genes({"sample": quant}, mapping)
    assert counts.loc["g1", "sample"] == 6.0
    assert counts.loc["g2", "sample"] == 7.0
    assert tpm.loc["g1", "sample"] == 25.0
    assert tpm.loc["g2", "sample"] == 75.0


def test_aggregate_rejects_unmapped_transcript():
    quant = pd.DataFrame({"NumReads": [1.0], "TPM": [1.0]}, index=["missing"])
    mapping = pd.DataFrame({"transcript_id": ["tx1"], "gene_id": ["g1"]})
    with pytest.raises(ValueError, match="lack a gene mapping"):
        aggregate_to_genes({"sample": quant}, mapping)


def test_log2_cpm_columns_have_common_total_scale():
    values = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 6.0]}, index=["g1", "g2"])
    result = log2_cpm(values, pseudocount=0.0)
    assert np.allclose(result["a"], result["b"])


def test_read_salmon_quant_validates_metadata(tmp_path):
    sample = tmp_path / "sample"
    (sample / "aux_info").mkdir(parents=True)
    pd.DataFrame(
        [["tx1", 100, 80.0, 1_000_000.0, 8.0]], columns=["Name", "Length", "EffectiveLength", "TPM", "NumReads"]
    ).to_csv(sample / "quant.sf", sep="\t", index=False)
    (sample / "aux_info/meta_info.json").write_text(
        json.dumps({"quant_errors": [], "num_valid_targets": 1, "num_mapped": 8}), encoding="utf-8"
    )
    (sample / "lib_format_counts.json").write_text(
        json.dumps({"expected_format": "ISR", "num_assigned_fragments": 8}), encoding="utf-8"
    )
    quant, details = read_salmon_quant(sample)
    assert quant.index.tolist() == ["tx1"]
    assert details["library"]["expected_format"] == "ISR"


def test_direction_agreement_excludes_small_changes_in_both_methods():
    first = pd.Series([0.01, 0.4, -0.6, 0.8], index=list("abcd"))
    second = pd.Series([-0.02, 0.5, -0.7, -0.9], index=list("abcd"))
    n, fraction = direction_agreement(first, second, minimum_abs_change=0.25)
    assert n == 3
    assert fraction == pytest.approx(2 / 3)


def test_direction_agreement_rejects_misaligned_genes():
    with pytest.raises(ValueError, match="identically ordered"):
        direction_agreement(pd.Series([1], index=["g1"]), pd.Series([1], index=["g2"]))
