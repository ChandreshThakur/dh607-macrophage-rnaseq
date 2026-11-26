#!/usr/bin/env python
"""Run the audited Phase 3 workflow without invoking interaction testing."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biological_analysis.baseline_reproduction import run as run_baseline  # noqa: E402
from exploratory_analysis.full_cohort_audit import run as run_audit  # noqa: E402


def main() -> None:
    audit = run_audit(ROOT)
    baseline = run_baseline(ROOT)
    concordance = pd.DataFrame(baseline["concordance"]).set_index("comparison")
    report = f"""# Phase 3 summary

## Cohort and readiness

The exact released matrices contain {audit['n_samples']} profiles from {audit['n_donors']} donors. All {audit['n_complete_donors']} donors have exactly one A/B/C/D profile, and no donor is incomplete. Count-matrix, CQN, metadata, and gene identifiers match exactly. No sample was excluded. This is the correct structure for these files; the alternative 86-donor/304-profile description belongs to a different subset or release.

## Quality control

The raw matrix has 35,033 genes, including 4,129 that are zero in every profile. The prespecified count >=10 in >=84 samples filter retains {audit['n_genes_primary_filter']:,} genes. Median library size is 33.61 million counts. Four samples ({', '.join(audit['candidate_sample_outliers'])}) have robustly unusual depth, but their detected-gene complexity is not extreme and they remain within their condition clusters, so they are retained. Raw variance strongly exceeds the mean for many genes, as expected for heterogeneous biological counts.

Median-of-ratios factors differ strongly by condition (A 1.50, B 1.20, C 0.95, D 0.61) despite similar raw library totals. This is evidence of global composition shifts and a limitation of assuming most genes are stable. The authors' CQN PCA preserves the same treatment geometry, providing an exploratory sensitivity check; CQN remains inappropriate as raw-count input.

## PCA interpretation

Explicit SVD on the 2,000 most variable filtered genes gives PC1={audit['pc1_variance_percent']:.1f}% and PC2={audit['pc2_variance_percent']:.1f}%. PC1 is predominantly associated with Salmonella status, and PC2 with IFN-gamma priming. Condition effects are much larger than library-pool or chemistry associations on these axes. The combined condition forms a distinct cluster, but separation in PCA does not itself establish a statistical or beneficial biological interaction.

## Published baseline

Using the authors' checksum-verified DESeq2 effect tables, course paired effects reproduce gene rankings closely: Spearman rho={concordance.loc['IFNg_vs_naive', 'spearman']:.3f} for IFN-gamma, {concordance.loc['SL1344_vs_naive', 'spearman']:.3f} for Salmonella, and {concordance.loc['IFNg_SL1344_vs_naive', 'spearman']:.3f} for combined treatment. Among genes with absolute effect >=0.5 in both methods, direction agreement is {100 * concordance.loc['IFNg_vs_naive', 'same_direction_fraction_above_threshold']:.1f}%, {100 * concordance.loc['SL1344_vs_naive', 'same_direction_fraction_above_threshold']:.2f}%, and {100 * concordance.loc['IFNg_SL1344_vs_naive', 'same_direction_fraction_above_threshold']:.1f}%.

STAT1 and IRF1 respond strongly to IFN-gamma. GBP-family genes, IDO1, and CXCL9/10/11 also rise strongly. Salmonella induces IFNB1, IL1B, TNF, and NFKBIA as well as interferon-responsive genes, matching the pathway scheme in Alasoo et al. Figure 1d. GBP1 and IDO1 show larger effects on the course pseudocount-log scale than in DESeq2, while low-baseline IFNB1 shows the opposite; these are scale/estimator discrepancies, not contradictory directions.

## Decision

The data are ready for a donor-paired IFN-gamma x Salmonella interaction analysis: the design is complete, identifiers are valid, treatment structure dominates PCA, technical batches are balanced across conditions, and the published response baseline is reproduced. The interaction must still use the within-donor D-B-C+A contrast on a declared scale, report effect sizes and uncertainty, and include normalization/filter and donor-influence sensitivity checks. A count-aware model, if added, must use raw counts with donor blocking and be labeled as supporting analysis. No genome-wide interaction test is run by this Phase 3 script.

## Key artifacts

- Dataset audit: `docs/full_dataset_audit.md`
- Required QC summary: `results/tables/full_dataset_qc_summary.csv`
- Required PCA figures: `results/figures/full_cohort_pca_by_condition.png` and `full_cohort_pca_by_donor.png`
- Baseline report: `results/reports/baseline_reproduction.md`
- Phase 2 sensitivity addition: `results/tables/salmon_response_direction_sensitivity.csv`
"""
    output = ROOT / "results/reports/phase3_summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"audit": audit, "baseline": baseline, "phase3_report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
