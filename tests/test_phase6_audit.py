from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reporting.phase6_audit import audit_cohort, audit_interaction_counts


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "results/tables/interaction_model_comparison.csv").is_file(),
    reason="Run scripts/run_phase4.py with the published inputs first",
)
def test_phase6_recomputed_core_counts():
    cohort = audit_cohort(ROOT)
    interaction = audit_interaction_counts(ROOT)
    assert cohort == {
        "samples": 336,
        "donors": 84,
        "complete_factorial_donors": 84,
        "genes_in_raw_matrix": 35033,
    }
    assert interaction["tested_genes"] == 15377
    assert interaction["primary_fdr_significant"] == 12765
    assert interaction["count_unreliable"] == 57
    assert interaction["primary_significant_among_count_reliable"] == 12720
    assert interaction["both_significant_same_direction_count_reliable"] == 12033
    assert interaction["robust_material_interactions"] == 6477
    assert interaction["robust_material_positive"] == 3479
    assert interaction["robust_material_negative"] == 2998


@pytest.mark.skipif(
    not (ROOT / "results/tables/phase6_audit_summary.json").is_file(),
    reason="Run scripts/run_phase6.py first",
)
def test_phase6_read_decision_and_required_outputs():
    summary = json.loads((ROOT / "results/tables/phase6_audit_summary.json").read_text())
    reads = summary["read_validation"]
    assert reads["ena_project"] == "PRJEB18997"
    assert reads["analyzed_cohort_public_donors"] == 33
    assert reads["analyzed_cohort_public_libraries"] == 132
    assert reads["planned_subset_donors"] == 12
    assert reads["planned_subset_libraries"] == 48
    assert reads["planned_subset_fastq_gib"] > summary["resources"]["machine"]["free_disk_gib"]
    assert reads["raw_reads_downloaded"] is False

    subset = pd.read_csv(ROOT / "data/manifests/ifnl_read_validation_subset_manifest.tsv", sep="\t")
    assert subset.donor.nunique() == 12
    assert subset.sample_alias.nunique() == 48
    assert set(subset.condition) == set("ABCD")

    required = [
        "results/tables/software_versions.csv",
        "data/manifests/final_output_checksums.csv",
        "presentation/DH607_final_presentation_updated.pptx",
    ]
    for relative in required:
        path = ROOT / relative
        assert path.exists() and path.stat().st_size > 0, relative


@pytest.mark.skipif(
    not (ROOT / "results/tables/phase6_audit_summary.json").is_file(),
    reason="Run scripts/run_phase6.py first",
)
def test_final_checksum_manifest_matches_files():
    manifest = pd.read_csv(ROOT / "data/manifests/final_output_checksums.csv")
    assert manifest.relative_path.is_unique
    required = {
        "data/manifests/ena_prjeb18997_run_manifest.tsv",
        "data/manifests/ifnl_read_validation_subset_manifest.tsv",
        "results/tables/phase6_audit_summary.json",
        "results/tables/software_versions.csv",
        "presentation/DH607_final_presentation_updated.pptx",
    }
    assert required.issubset(manifest.relative_path)
    for row in manifest.itertuples(index=False):
        path = ROOT / row.relative_path
        assert path.exists()
        assert path.stat().st_size == row.size_bytes
