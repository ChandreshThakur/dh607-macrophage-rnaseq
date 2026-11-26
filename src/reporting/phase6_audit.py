"""Phase 6 final audit and reproducibility-manifest generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


FDR = 0.05
MATERIAL_EFFECT = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sum_semicolon_integers(value: object) -> int:
    if pd.isna(value):
        return 0
    return sum(int(item) for item in str(value).split(";") if item.strip())


def _condition_from_alias(alias: str) -> str:
    lower = alias.lower()
    if "with salmonella and ifng" in lower:
        return "D"
    if "with salmonella" in lower:
        return "C"
    if "with ifng" in lower:
        return "B"
    if "untreated" in lower:
        return "A"
    raise ValueError(f"Unrecognized ENA RNA sample alias: {alias}")


def audit_interaction_counts(root: Path) -> dict[str, int | float]:
    comparison = pd.read_csv(root / "results/tables/interaction_model_comparison.csv")
    reliable = comparison["count_reliable"].astype(bool)
    same = comparison["same_direction"].astype(bool)
    primary = comparison["course_adjusted_p_value"] < FDR
    count = comparison["count_adjusted_p_value"] < FDR
    material = comparison["course_interaction_log2"].abs() >= MATERIAL_EFFECT
    strict = reliable & same & primary & count
    robust = strict & material
    return {
        "tested_genes": int(len(comparison)),
        "primary_fdr_significant": int(primary.sum()),
        "primary_fdr_fraction": float(primary.mean()),
        "count_unreliable": int((~reliable).sum()),
        "primary_significant_among_count_reliable": int((primary & reliable).sum()),
        "count_significant_among_count_reliable": int((count & reliable).sum()),
        "both_significant_same_direction_count_reliable": int(strict.sum()),
        "robust_material_interactions": int(robust.sum()),
        "robust_material_positive": int((robust & (comparison.course_interaction_log2 > 0)).sum()),
        "robust_material_negative": int((robust & (comparison.course_interaction_log2 < 0)).sum()),
    }


def audit_cohort(root: Path) -> dict[str, int]:
    metadata = pd.read_csv(root / "data/published/RNA_sample_metadata.txt.gz", sep="\t")
    counts = pd.read_csv(root / "data/published/RNA_count_matrix.txt.gz", sep="\t", index_col=0)
    per_donor = metadata.groupby("donor")["condition"].agg(lambda x: "".join(sorted(x)))
    if list(counts.columns) != list(metadata.sample_id):
        raise ValueError("Count matrix and metadata sample order differ")
    return {
        "samples": int(len(metadata)),
        "donors": int(metadata.donor.nunique()),
        "complete_factorial_donors": int((per_donor == "ABCD").sum()),
        "genes_in_raw_matrix": int(len(counts)),
    }


def build_read_validation_manifest(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit official ENA metadata and choose a donor-balanced technical subset.

    This writes metadata only. It never downloads sequencing reads.
    """
    ena_path = root / "data/manifests/ena_prjeb18997_run_manifest.tsv"
    ena = pd.read_csv(ena_path, sep="\t")
    metadata = pd.read_csv(root / "data/published/RNA_sample_metadata.txt.gz", sep="\t")
    ifnl_donor = pd.read_csv(root / "results/tables/ifnl_donor_validation.csv")
    ena["fastq_bytes_total"] = ena.fastq_bytes.map(_sum_semicolon_integers)
    ena["submitted_bytes_total"] = ena.submitted_bytes.map(_sum_semicolon_integers)
    rna = ena.loc[ena.library_strategy.eq("RNA-Seq")].copy()

    donor_map = metadata[["donor", "genotype_id", "open_access"]].drop_duplicates()
    genotype_to_donor = donor_map.set_index("genotype_id")["donor"].to_dict()

    def match_genotype(alias: str) -> str | None:
        hits = [genotype for genotype in genotype_to_donor if alias.startswith(genotype)]
        if len(hits) > 1:
            raise ValueError(f"Ambiguous genotype prefix for {alias}")
        return hits[0] if hits else None

    rna["genotype_id"] = rna.sample_alias.map(match_genotype)
    cohort_rna = rna.loc[rna.genotype_id.notna()].copy()
    cohort_rna["donor"] = cohort_rna.genotype_id.map(genotype_to_donor)
    cohort_rna["condition"] = cohort_rna.sample_alias.map(_condition_from_alias)
    sample_conditions = cohort_rna[["sample_alias", "donor", "condition"]].drop_duplicates()
    if len(sample_conditions) != 132 or sample_conditions.donor.nunique() != 33:
        raise ValueError("Expected public reads for 33 analyzed donors and 132 libraries")
    completeness = sample_conditions.groupby("donor").condition.agg(lambda x: "".join(sorted(x)))
    if not (completeness == "ABCD").all():
        raise ValueError("Public ENA donor set is not a complete four-condition subset")

    open_donors = set(donor_map.loc[donor_map.open_access.eq(1), "donor"])
    effects = ifnl_donor.loc[ifnl_donor.gene_name.eq("IFNL1"), ["donor", "interaction_log2"]]
    effects = effects.loc[effects.donor.isin(open_donors)].sort_values("interaction_log2").reset_index(drop=True)
    indices = np.unique(np.rint(np.linspace(0, len(effects) - 1, 12)).astype(int))
    selected_donors = effects.iloc[indices].copy()
    if len(selected_donors) != 12:
        raise ValueError("Could not select 12 distinct public donors")
    subset = cohort_rna.loc[cohort_rna.donor.isin(selected_donors.donor)].copy()
    subset = subset.merge(selected_donors, on="donor", how="left")
    subset = subset.sort_values(["interaction_log2", "donor", "condition", "run_accession"])
    if subset.sample_alias.nunique() != 48 or subset.donor.nunique() != 12:
        raise ValueError("Expected 12 donors and 48 libraries in the proposed subset")

    out_columns = [
        "donor", "condition", "interaction_log2", "genotype_id", "sample_alias",
        "sample_accession", "run_accession", "run_alias", "read_count", "base_count",
        "fastq_ftp", "fastq_bytes", "fastq_md5", "fastq_bytes_total",
        "submitted_ftp", "submitted_bytes", "submitted_md5", "submitted_bytes_total",
    ]
    subset[out_columns].to_csv(
        root / "data/manifests/ifnl_read_validation_subset_manifest.tsv", sep="\t", index=False
    )

    def gib(value: int) -> float:
        return float(value / 2**30)

    public_fastq = int(cohort_rna.fastq_bytes_total.sum())
    public_cram = int(cohort_rna.submitted_bytes_total.sum())
    subset_fastq = int(subset.fastq_bytes_total.sum())
    subset_cram = int(subset.submitted_bytes_total.sum())
    summary: dict[str, object] = {
        "ena_project": "PRJEB18997",
        "ena_study": "ERP020977",
        "ena_manifest_runs_all_assays": int(len(ena)),
        "ena_rna_runs": int(len(rna)),
        "ena_rna_libraries": int(rna.sample_alias.nunique()),
        "analyzed_cohort_public_donors": int(cohort_rna.donor.nunique()),
        "analyzed_cohort_public_libraries": int(cohort_rna.sample_alias.nunique()),
        "analyzed_cohort_public_runs": int(len(cohort_rna)),
        "analyzed_cohort_fastq_bytes": public_fastq,
        "analyzed_cohort_fastq_gib": gib(public_fastq),
        "analyzed_cohort_submitted_cram_bytes": public_cram,
        "analyzed_cohort_submitted_cram_gib": gib(public_cram),
        "planned_subset_donors": int(subset.donor.nunique()),
        "planned_subset_libraries": int(subset.sample_alias.nunique()),
        "planned_subset_runs": int(len(subset)),
        "planned_subset_fastq_bytes": subset_fastq,
        "planned_subset_fastq_gib": gib(subset_fastq),
        "planned_subset_submitted_cram_bytes": subset_cram,
        "planned_subset_submitted_cram_gib": gib(subset_cram),
        "planned_subset_workspace_gib_input_plus_3x_working": gib(subset_fastq) * 4,
        "raw_reads_downloaded": False,
    }
    return subset[out_columns], summary


def write_software_versions(root: Path) -> pd.DataFrame:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "pydeseq2", "pytest"]
    rows = [{"component": "Python", "version": platform.python_version(), "source": "runtime"}]
    for package in packages:
        rows.append({
            "component": package,
            "version": importlib.metadata.version(package),
            "source": "installed Python environment",
        })
    index_metadata = root / "data/reference/salmon_index_ensembl87/versionInfo.json"
    if index_metadata.is_file():
        index_info = json.loads(index_metadata.read_text(encoding="utf-8"))
        rows.append({"component": "Salmon", "version": index_info["salmonVersion"], "source": "saved index metadata"})
    rows.extend([
        {"component": "FastQC", "version": "0.12.1", "source": "environment.yml pin for optional pilot"},
    ])
    frame = pd.DataFrame(rows).drop_duplicates(["component", "version"])
    frame.to_csv(root / "results/tables/software_versions.csv", index=False)
    return frame


def write_final_checksums(root: Path) -> pd.DataFrame:
    include = [
        "configs/analysis.json", "configs/phase6_resources.json", "environment.yml", "pyproject.toml",
        "data/manifests/ena_prjeb18997_run_manifest.tsv",
        "data/manifests/ifnl_read_validation_subset_manifest.tsv",
        "results/tables/phase6_audit_summary.json", "results/tables/software_versions.csv",
        "docs/phase4_statistical_design.md", "docs/phase4_biological_hypotheses.md",
        "presentation/DH607_final_presentation_updated.pptx",
        "src/reporting/phase6_audit.py", "scripts/run_phase6.py", "tests/test_phase6_audit.py",
    ]
    rows = []
    for relative in include:
        path = root / relative
        if not path.exists():
            continue
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "data/manifests/final_output_checksums.csv", index=False)
    return frame


def run(root: Path) -> dict[str, object]:
    interaction = audit_interaction_counts(root)
    cohort = audit_cohort(root)
    _, reads = build_read_validation_manifest(root)
    resources = json.loads((root / "configs/phase6_resources.json").read_text(encoding="utf-8"))
    write_software_versions(root)
    summary: dict[str, object] = {
        "audit_date": "2026-10-27",
        "cohort": cohort,
        "interaction": interaction,
        "read_validation": reads,
        "resources": resources,
    }
    output = root / "results/tables/phase6_audit_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_final_checksums(root)
    return summary
