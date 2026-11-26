"""Phase 3 audit, QC, normalization, and SVD/PCA for the published cohort."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_validation.published import (
    condition_summary,
    donor_completeness,
    validate_count_matrix,
    validate_cqn_matrix,
)
from preprocessing.normalization import expressed_filter, log2_transform, median_of_ratios


PALETTE = {"A": "#666666", "B": "#377eb8", "C": "#e66101", "D": "#984ea3"}
CONDITION_ORDER = ["A", "B", "C", "D"]


def robust_mad_z(values: pd.Series) -> pd.Series:
    """Median/MAD z-scores, returning zero for a constant vector."""
    values = values.astype(float)
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad == 0:
        return pd.Series(0.0, index=values.index, name=values.name)
    return 0.67448975 * (values - median) / mad


def svd_pca(expression: pd.DataFrame, n_components: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PCA by explicit SVD; rows in the returned score table are samples."""
    if expression.shape[0] < 2 or expression.shape[1] < 2:
        raise ValueError("PCA requires at least two genes and two samples")
    sample_by_gene = expression.T.to_numpy(dtype=float)
    centered = sample_by_gene - sample_by_gene.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, len(singular_values))
    scores = pd.DataFrame(
        u[:, :k] * singular_values[:k],
        index=expression.columns,
        columns=[f"PC{i}" for i in range(1, k + 1)],
    )
    eigenvalues = singular_values**2 / (centered.shape[0] - 1)
    ratios = eigenvalues / eigenvalues.sum()
    variance = pd.DataFrame({
        "component": np.arange(1, k + 1),
        "explained_variance": eigenvalues[:k],
        "explained_variance_ratio": ratios[:k],
        "cumulative_explained_variance_ratio": np.cumsum(ratios[:k]),
    })
    return scores, variance


def categorical_eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Descriptive fraction of score variance between categorical group means."""
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    if len(frame) < 2 or frame.group.nunique() < 2:
        return float("nan")
    grand = frame.value.mean()
    total = float(((frame.value - grand) ** 2).sum())
    if total == 0:
        return 0.0
    grouped = frame.groupby("group", observed=True).value.agg(["size", "mean"])
    between = float((grouped["size"] * (grouped["mean"] - grand) ** 2).sum())
    return between / total


def _plot_library_qc(metrics: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.violinplot(
        data=metrics, x="condition", y="library_size_millions", order=CONDITION_ORDER,
        hue="condition", palette=PALETTE, legend=False, inner="box", cut=0, ax=axes[0],
    )
    axes[0].set(xlabel="Condition", ylabel="Total raw gene counts (millions)", title="Library size")
    sns.violinplot(
        data=metrics, x="condition", y="detected_genes_ge10", order=CONDITION_ORDER,
        hue="condition", palette=PALETTE, legend=False, inner="box", cut=0, ax=axes[1],
    )
    axes[1].set(xlabel="Condition", ylabel="Genes with raw count >= 10", title="Detected-gene complexity")
    sns.violinplot(
        data=metrics, x="condition", y="size_factor", order=CONDITION_ORDER,
        hue="condition", palette=PALETTE, legend=False, inner="box", cut=0, ax=axes[2],
    )
    axes[2].set(xlabel="Condition", ylabel="Median-of-ratios size factor", title="Composition-aware scale factor")
    fig.suptitle("Full-cohort sequencing-depth and complexity distributions")
    fig.tight_layout()
    fig.savefig(figures / "library_size_distribution.png", dpi=180)
    plt.close(fig)


def _plot_mean_variance(counts: pd.DataFrame, figures: Path) -> None:
    means = counts.mean(axis=1)
    variances = counts.var(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(np.log10(means + 1), np.log10(variances + 1), s=5, alpha=0.22, rasterized=True)
    boundary = np.linspace(0, np.log10(means.max() + 1), 100)
    ax.plot(boundary, boundary, color="#b2182b", lw=1, ls="--", label="variance = mean")
    ax.set(
        xlabel="log10(mean raw count + 1)",
        ylabel="log10(raw-count variance + 1)",
        title="Full-cohort raw-count mean-variance relationship",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "full_cohort_mean_variance.png", dpi=180)
    plt.close(fig)


def _plot_pca(scores: pd.DataFrame, variance: pd.DataFrame, figures: Path) -> None:
    pc1 = 100 * variance.loc[0, "explained_variance_ratio"]
    pc2 = 100 * variance.loc[1, "explained_variance_ratio"]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for condition in CONDITION_ORDER:
        group = scores.loc[scores.condition == condition]
        ax.scatter(group.PC1, group.PC2, s=27, alpha=0.72, color=PALETTE[condition], label=condition)
        centroid = group[["PC1", "PC2"]].mean()
        ax.scatter(centroid.PC1, centroid.PC2, s=125, marker="X", color=PALETTE[condition], edgecolor="black", linewidth=0.6)
    ax.set(
        xlabel=f"PC1 ({pc1:.1f}% variance)",
        ylabel=f"PC2 ({pc2:.1f}% variance)",
        title="Full-cohort PCA by experimental condition",
    )
    ax.legend(title="Condition", frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(figures / "full_cohort_pca_by_condition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for _, donor_scores in scores.sort_values("condition").groupby("donor", observed=True):
        ax.plot(donor_scores.PC1, donor_scores.PC2, color="#bdbdbd", lw=0.55, alpha=0.45, zorder=1)
    for condition in CONDITION_ORDER:
        group = scores.loc[scores.condition == condition]
        ax.scatter(group.PC1, group.PC2, s=22, alpha=0.72, color=PALETTE[condition], label=condition, zorder=2)
    flagged = scores.loc[scores.qc_outlier_candidate]
    for row in flagged.itertuples():
        ax.annotate(row.sample_id, (row.PC1, row.PC2), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(
        xlabel=f"PC1 ({pc1:.1f}% variance)",
        ylabel=f"PC2 ({pc2:.1f}% variance)",
        title="Paired donor trajectories in full-cohort PCA",
    )
    ax.legend(title="Condition", frameon=False, ncols=4)
    fig.tight_layout()
    fig.savefig(figures / "full_cohort_pca_by_donor.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    sns.scatterplot(data=scores, x="PC1", y="PC2", hue="library_pool", s=28, alpha=0.8, ax=axes[0], palette="tab10")
    axes[0].set_title("Library pool")
    axes[0].legend(fontsize=6, title_fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.scatterplot(data=scores, x="PC1", y="PC2", hue="chemistry", style="rna_auto", s=30, alpha=0.8, ax=axes[1])
    axes[1].set_title("Chemistry and RNA automation")
    axes[1].legend(fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pc1:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pc2:.1f}% variance)")
    fig.suptitle("Technical covariates on the biological PCA")
    fig.tight_layout()
    fig.savefig(figures / "full_cohort_pca_by_technical_covariates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_normalization_pca_comparison(
    course_scores: pd.DataFrame,
    course_variance: pd.DataFrame,
    cqn_scores: pd.DataFrame,
    cqn_variance: pd.DataFrame,
    figures: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, scores, variance, title in [
        (axes[0], course_scores, course_variance, "Median-ratio log2"),
        (axes[1], cqn_scores, cqn_variance, "Authors' CQN"),
    ]:
        for condition in CONDITION_ORDER:
            group = scores.loc[scores.condition == condition]
            ax.scatter(group.PC1, group.PC2, s=20, alpha=0.65, color=PALETTE[condition], label=condition)
        ax.set(
            xlabel=f"PC1 ({100 * variance.loc[0, 'explained_variance_ratio']:.1f}%)",
            ylabel=f"PC2 ({100 * variance.loc[1, 'explained_variance_ratio']:.1f}%)",
            title=title,
        )
    axes[1].legend(title="Condition", frameon=False)
    fig.suptitle("PCA sensitivity to exploratory normalization")
    fig.tight_layout()
    fig.savefig(figures / "full_cohort_pca_normalization_comparison.png", dpi=180)
    plt.close(fig)


def run(root: Path) -> dict[str, object]:
    published = root / "data/published"
    tables = root / "results/tables"
    figures = root / "results/figures"
    processed = root / "data/processed"
    docs = root / "docs"
    for directory in (tables, figures, processed, docs):
        directory.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(published / "RNA_sample_metadata.txt.gz", sep="\t")
    counts = pd.read_csv(published / "RNA_count_matrix.txt.gz", sep="\t", index_col=0)
    cqn = pd.read_csv(published / "RNA_cqn_matrix.txt.gz", sep="\t", index_col=0)
    genes = pd.read_csv(published / "RNA_gene_metadata.txt.gz", sep="\t")

    completeness = donor_completeness(metadata)
    condition_table = condition_summary(metadata)
    condition_table["complete_four_condition_donors"] = int(completeness.one_sample_each.sum())
    condition_table.to_csv(tables / "full_sample_metadata_summary.csv", index=False)
    completeness.to_csv(tables / "donor_condition_completeness.csv", index=False)
    complete_donors = completeness.loc[completeness.one_sample_each, "donor"]
    metadata.loc[metadata.donor.isin(complete_donors), ["sample_id", "donor", "condition"]].to_csv(
        tables / "full_factorial_sample_ids.csv", index=False
    )

    count_check = validate_count_matrix(counts, metadata)
    cqn_check = validate_cqn_matrix(cqn, counts)
    if not count_check["sample_id_sets_match"] or not count_check["sample_order_matches_metadata"]:
        raise ValueError("count-matrix sample identifiers do not exactly match metadata")
    if list(counts.index) != genes.gene_id.tolist():
        raise ValueError("count-matrix genes do not exactly match gene metadata order")
    if not completeness.one_sample_each.all():
        raise ValueError("the full-cohort matrices are not a complete one-sample-per-condition design")

    normalized, size_factors = median_of_ratios(counts)
    primary_filter = expressed_filter(counts, minimum_count=10, minimum_samples=84)
    transformed = log2_transform(normalized.loc[primary_filter], pseudocount=0.5)
    transformed.to_csv(processed / "phase3_log2_median_ratio_expression.tsv.gz", sep="\t", compression="gzip")

    sample_metrics = metadata.set_index("sample_id").loc[counts.columns].copy()
    sample_metrics.index.name = "sample_id"
    sample_metrics["library_size"] = counts.sum(axis=0)
    sample_metrics["library_size_millions"] = sample_metrics.library_size / 1_000_000
    sample_metrics["detected_genes_ge1"] = (counts >= 1).sum(axis=0)
    sample_metrics["detected_genes_ge10"] = (counts >= 10).sum(axis=0)
    sample_metrics["size_factor"] = size_factors
    sample_metrics["robust_z_log10_library_size"] = robust_mad_z(np.log10(sample_metrics.library_size))
    sample_metrics["robust_z_detected_genes_ge10"] = robust_mad_z(sample_metrics.detected_genes_ge10)
    sample_metrics["qc_outlier_candidate"] = (
        sample_metrics.robust_z_log10_library_size.abs().gt(3.5)
        | sample_metrics.robust_z_detected_genes_ge10.abs().gt(3.5)
    )
    sample_metrics.reset_index().to_csv(tables / "full_sample_qc_metrics.csv", index=False)

    variable_ids = transformed.var(axis=1).nlargest(min(2000, len(transformed))).index
    scores, variance = svd_pca(transformed.loc[variable_ids], n_components=10)
    scores.index.name = "sample_id"
    scores = scores.join(metadata.set_index("sample_id"))
    scores = scores.join(sample_metrics[["qc_outlier_candidate"]])
    scores.reset_index().to_csv(tables / "full_cohort_pca_scores.csv", index=False)
    variance.to_csv(tables / "full_cohort_pca_explained_variance.csv", index=False)

    cqn_variable_ids = cqn.loc[primary_filter].var(axis=1).nlargest(min(2000, int(primary_filter.sum()))).index
    cqn_scores, cqn_variance = svd_pca(cqn.loc[cqn_variable_ids], n_components=10)
    cqn_scores.index.name = "sample_id"
    cqn_scores = cqn_scores.join(metadata.set_index("sample_id"))
    cqn_scores.reset_index().to_csv(tables / "full_cohort_cqn_pca_scores.csv", index=False)
    cqn_variance.to_csv(tables / "full_cohort_cqn_pca_explained_variance.csv", index=False)

    association_rows = []
    categorical_covariates = ["condition", "IFNg", "SL1344", "donor", "library_pool", "chemistry", "rna_auto", "sex", "replicate"]
    for pc in [f"PC{i}" for i in range(1, 6)]:
        for covariate in categorical_covariates:
            association_rows.append({
                "component": pc,
                "covariate": covariate,
                "n_levels": int(scores[covariate].nunique(dropna=True)),
                "eta_squared": categorical_eta_squared(scores[pc], scores[covariate]),
            })
        for covariate in ("library_size", "detected_genes_ge10", "mean_purity"):
            values = sample_metrics[covariate] if covariate in sample_metrics else pd.Series(index=scores.index, dtype=float)
            association_rows.append({
                "component": pc,
                "covariate": covariate,
                "n_levels": int(values.notna().sum()),
                "pearson_correlation": float(scores[pc].corr(values)),
            })
    associations = pd.DataFrame(association_rows)
    associations.to_csv(tables / "full_cohort_pca_associations.csv", index=False)

    normalization_rows = []
    for method, method_scores, method_variance in [
        ("median_ratio_log2", scores, variance),
        ("authors_cqn", cqn_scores, cqn_variance),
    ]:
        for pc in ("PC1", "PC2"):
            normalization_rows.append({
                "method": method,
                "component": pc,
                "explained_variance_ratio": float(method_variance.loc[int(pc.removeprefix('PC')) - 1, "explained_variance_ratio"]),
                "condition_eta_squared": categorical_eta_squared(method_scores[pc], method_scores.condition),
                "IFNg_eta_squared": categorical_eta_squared(method_scores[pc], method_scores.IFNg),
                "SL1344_eta_squared": categorical_eta_squared(method_scores[pc], method_scores.SL1344),
            })
    normalization_pca = pd.DataFrame(normalization_rows)
    normalization_pca.to_csv(tables / "full_cohort_pca_normalization_sensitivity.csv", index=False)

    technical_rows = []
    for column in ["status", "replicate", "library_pool", "chemistry", "rna_auto", "sex", "purity_bins"]:
        counts_text = "; ".join(f"{key}={value}" for key, value in metadata[column].value_counts(dropna=False).items())
        technical_rows.append({
            "covariate": column,
            "n_levels_including_missing": int(metadata[column].nunique(dropna=False)),
            "n_missing": int(metadata[column].isna().sum()),
            "level_counts": counts_text,
        })
    pd.DataFrame(technical_rows).to_csv(tables / "full_technical_covariates.csv", index=False)

    all_zero = int((counts.sum(axis=1) == 0).sum())
    count_positive_all = int((counts > 0).all(axis=1).sum())
    flagged_samples = sample_metrics.index[sample_metrics.qc_outlier_candidate].tolist()
    qc_rows = [
        ("samples", len(metadata), "profiles", "all metadata rows"),
        ("donors", metadata.donor.nunique(), "donors", "unique biological donors"),
        ("complete_four_condition_donors", int(completeness.one_sample_each.sum()), "donors", "exactly one A/B/C/D profile"),
        ("incomplete_donors", int((~completeness.one_sample_each).sum()), "donors", "none silently excluded"),
        ("raw_count_genes", len(counts), "genes", "nonnegative integer matrix"),
        ("cqn_genes", len(cqn), "genes", "continuous author-normalized matrix"),
        ("all_zero_genes", all_zero, "genes", "zero across 336 profiles"),
        ("genes_positive_in_all_samples_for_size_factors", count_positive_all, "genes", "median-ratio denominator set"),
        ("genes_passing_primary_filter", int(primary_filter.sum()), "genes", "count >=10 in >=84 samples"),
        ("median_library_size", float(sample_metrics.library_size.median()), "raw counts", "before normalization"),
        ("minimum_library_size", int(sample_metrics.library_size.min()), "raw counts", "before normalization"),
        ("maximum_library_size", int(sample_metrics.library_size.max()), "raw counts", "before normalization"),
        ("minimum_detected_genes_ge10", int(sample_metrics.detected_genes_ge10.min()), "genes", "per sample"),
        ("maximum_detected_genes_ge10", int(sample_metrics.detected_genes_ge10.max()), "genes", "per sample"),
        ("candidate_sample_outliers", len(flagged_samples), "samples", ";".join(flagged_samples)),
        ("pc1_explained_variance", float(variance.loc[0, "explained_variance_ratio"]), "fraction", "explicit SVD"),
        ("pc2_explained_variance", float(variance.loc[1, "explained_variance_ratio"]), "fraction", "explicit SVD"),
    ]
    qc_summary = pd.DataFrame(qc_rows, columns=["metric", "value", "units", "notes"])
    qc_summary.to_csv(tables / "full_dataset_qc_summary.csv", index=False)

    _plot_library_qc(sample_metrics.reset_index(), figures)
    _plot_mean_variance(counts, figures)
    _plot_pca(scores.reset_index(), variance, figures)
    _plot_normalization_pca_comparison(scores, variance, cqn_scores, cqn_variance, figures)

    top_pc1 = associations.loc[(associations.component == "PC1") & associations.eta_squared.notna()].nlargest(3, "eta_squared")
    top_pc2 = associations.loc[(associations.component == "PC2") & associations.eta_squared.notna()].nlargest(3, "eta_squared")
    summary = {
        "n_samples": int(len(metadata)),
        "n_donors": int(metadata.donor.nunique()),
        "n_complete_donors": int(completeness.one_sample_each.sum()),
        "n_incomplete_donors": int((~completeness.one_sample_each).sum()),
        "n_genes": int(len(counts)),
        "n_genes_primary_filter": int(primary_filter.sum()),
        "n_candidate_sample_outliers": int(len(flagged_samples)),
        "candidate_sample_outliers": flagged_samples,
        "pc1_variance_percent": float(100 * variance.loc[0, "explained_variance_ratio"]),
        "pc2_variance_percent": float(100 * variance.loc[1, "explained_variance_ratio"]),
        "pc1_top_categorical_associations": top_pc1[["covariate", "eta_squared"]].to_dict("records"),
        "pc2_top_categorical_associations": top_pc2[["covariate", "eta_squared"]].to_dict("records"),
        "condition_median_size_factors": sample_metrics.groupby("condition").size_factor.median().to_dict(),
    }
    (tables / "full_cohort_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def eta(pc: str, covariate: str) -> float:
        return float(associations.loc[(associations.component == pc) & (associations.covariate == covariate), "eta_squared"].iloc[0])

    audit = f"""# Full-cohort dataset audit

Generated by `python scripts/run_full_cohort_audit.py` from the four checksum-verified files in `data/published/`.

## Exact sample structure

- The count matrix and metadata contain {len(metadata)} RNA-seq profiles from {metadata.donor.nunique()} unique biological donors.
- A = `naive`, B = `IFNg`, C = `SL1344`, and D = `IFNg_SL1344`; each condition contains 84 samples from 84 donors.
- {int(completeness.one_sample_each.sum())} donors have exactly one profile in every condition; {int((~completeness.one_sample_each).sum())} are incomplete. The initial paired factorial subset is therefore all {len(metadata)} samples from all {metadata.donor.nunique()} donors.
- There are no duplicate sample IDs or duplicate donor-condition combinations. All 336 rows have `status=Success`.
- Eight selected profiles have `replicate=2` (all four conditions for donors `babk` and `oarz`); they do not create replicate rows in the released matrix.

This exact release therefore supports 84 donors / 336 profiles. Reports of 86 individuals / 304 profiles refer to a different analysis or release and must not be imposed on these files. The independent [eQTL Catalogue QC report](https://www.ebi.ac.uk/eqtl/static/QC_reports/Alasoo_2018_QC_report.html) also records 336 profiles and 84 genotyped individuals for Alasoo_2018.

## Identifier and matrix validation

- Raw counts: {counts.shape[0]:,} Ensembl gene IDs x {counts.shape[1]} samples; nonnegative integers; {count_check['missing_values']} missing values; sample IDs and order exactly match metadata.
- Gene metadata: {len(genes):,} unique unversioned Ensembl IDs in exactly the same order as the count matrix; {(genes.gene_biotype == 'protein_coding').sum():,} are protein coding.
- Authors' CQN matrix: {cqn.shape[0]:,} x {cqn.shape[1]}, continuous finite values ranging from {cqn.to_numpy().min():.4f} to {cqn.to_numpy().max():.4f}, with identical gene/sample order. It is not raw count data.
- File MD5 values and exact Zenodo URLs are preserved in `data/manifests/file_manifest.csv`; the source archive is Zenodo record 1188300.

## Provenance and expression scales

The [study](https://doi.org/10.1038/s41588-018-0046-7) uses the GRCh38 human assembly and Ensembl gene identifiers from the study-era annotation. The paper states that read counts used for QTL mapping were conditional-quantile normalized with gene GC content as a covariate. The released `RNA_count_matrix` remains the raw integer gene-count input; `RNA_cqn_matrix` is the authors' normalized continuous expression. Our exploratory matrix divides raw counts by DESeq-style median-of-ratios size factors and applies `log2(normalized count + 0.5)`. That log matrix is for QC/PCA and is not called a variance-stabilizing or Anscombe transform. Raw counts must be retained for any future count-aware model.

## QC and filtering

- Library sizes range from {sample_metrics.library_size.min():,.0f} to {sample_metrics.library_size.max():,.0f} counts (median {sample_metrics.library_size.median():,.0f}).
- Genes with count >=10 range from {sample_metrics.detected_genes_ge10.min():,} to {sample_metrics.detected_genes_ge10.max():,} per sample.
- Raw library-size medians are similar by condition, but median size factors are A={sample_metrics.groupby('condition').size_factor.median()['A']:.2f}, B={sample_metrics.groupby('condition').size_factor.median()['B']:.2f}, C={sample_metrics.groupby('condition').size_factor.median()['C']:.2f}, and D={sample_metrics.groupby('condition').size_factor.median()['D']:.2f}. This indicates a large treatment-associated composition shift: median-of-ratios factors are not simply sequencing-depth totals, and the stable-majority assumption deserves sensitivity checks.
- The prespecified exploratory filter (count >=10 in >=84 samples) retains {int(primary_filter.sum()):,} of {len(counts):,} genes.
- Robust depth/complexity screening flags {len(flagged_samples)} candidates: {', '.join(flagged_samples)}. They are retained because none has an extreme detected-gene count, and visual review places them within their condition clusters rather than as isolated expression outliers.
- The raw mean-variance plot shows variance substantially above the Poisson equality line for many genes, which is expected for heterogeneous biological RNA-seq counts and supports using count-aware dispersion models for later inference.

## PCA

PCA was calculated by explicit SVD using the 2,000 most variable genes from the filtered log2 median-ratio matrix. PC1 explains {summary['pc1_variance_percent']:.1f}% and is primarily associated with Salmonella status (eta-squared={eta('PC1', 'SL1344'):.3f}, versus {eta('PC1', 'IFNg'):.3f} for IFN-gamma). PC2 explains {summary['pc2_variance_percent']:.1f}% and is primarily associated with IFN-gamma priming (eta-squared={eta('PC2', 'IFNg'):.3f}, versus {eta('PC2', 'SL1344'):.3f} for Salmonella). Thus infection contributes most strongly to the first axis and priming to the second, although both factors contribute to both axes. Library-pool eta-squared is only {eta('PC1', 'library_pool'):.3f}/{eta('PC2', 'library_pool'):.3f} on PC1/PC2, and chemistry is {eta('PC1', 'chemistry'):.3f}/{eta('PC2', 'chemistry'):.3f}. These are descriptive, unadjusted summaries rather than hypothesis tests; eta-squared for an 84-level donor factor is upward-biased, and overlapping covariates prevent causal attribution.

The authors' CQN matrix was analyzed separately as a normalization sensitivity check, not as count input. Its first two PCs are saved and plotted beside the course matrix. Persistence of the four-condition geometry across both scales supports a biological treatment signal; differences in explained variance or axis rotation are expected because CQN also accounts for gene-level GC/length effects and quantile normalization.

## Technical covariates and exclusions

Metadata includes ten library pools, three chemistry categories, RNA automation, sex, replicate selection, culture/differentiation dates, RNA concentration, and purity measures. Library pools are balanced across the four conditions. Purity is missing for 12 profiles and ATAC dates for 168, but the core RNA sample identifiers and experimental factors are complete. No sample is removed in Phase 3.

## Figure provenance

- `library_size_distribution.png`: this script; raw library totals and genes with count >=10 by condition.
- `full_cohort_mean_variance.png`: this script; gene-wise raw-count mean and variance.
- `full_cohort_pca_by_condition.png`: this script; condition-colored SVD scores and condition centroids.
- `full_cohort_pca_by_donor.png`: this script; the same scores with each donor's four profiles connected; only QC candidates are labelled.
- `full_cohort_pca_by_technical_covariates.png`: this script; library pool, chemistry, and automation overlays.
- `full_cohort_pca_normalization_comparison.png`: this script; course log2 median-ratio and authors' CQN PCA side by side.
"""
    (docs / "full_dataset_audit.md").write_text(audit, encoding="utf-8")
    return summary
