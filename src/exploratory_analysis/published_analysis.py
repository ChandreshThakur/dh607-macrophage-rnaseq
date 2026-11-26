"""Executable full-cohort QC, PCA, interaction testing, and robustness checks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

from preprocessing.normalization import expressed_filter, log2_transform, median_of_ratios
from statistical_analysis.interaction import benjamini_hochberg, interaction_contrasts, test_interactions

PALETTE = {"A": "#777777", "B": "#3670b5", "C": "#d95f02", "D": "#7b3294"}
CANDIDATES = ["GBP1", "GBP2", "GBP3", "GBP4", "GBP5", "STAT1", "IRF1", "IDO1", "IL10", "SOCS3", "DUSP1"]


def _annotate(results: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    annotation = genes.set_index("gene_id")[["gene_name", "gene_biotype", "chr", "length"]]
    return results.join(annotation, how="left").reset_index(names="gene_id")


def _paired_effect(expression: pd.DataFrame, metadata: pd.DataFrame, treated: str) -> pd.DataFrame:
    lookup = metadata.set_index(["donor", "condition"])["sample_id"]
    donors = sorted(metadata.donor.unique())
    a = expression.loc[:, [lookup.loc[(d, "A")] for d in donors]].to_numpy()
    b = expression.loc[:, [lookup.loc[(d, treated)] for d in donors]].to_numpy()
    delta = b - a
    mean = delta.mean(axis=1)
    se = delta.std(axis=1, ddof=1) / np.sqrt(len(donors))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / se
    t = np.where((se == 0) & (mean == 0), 0.0, t)
    p = 2 * stats.t.sf(np.abs(t), len(donors) - 1)
    p = np.where((se == 0) & (mean == 0), 1.0, p)
    return pd.DataFrame({"mean_log2_change": mean, "p_value": p, "adjusted_p_value": benjamini_hochberg(p)}, index=expression.index)


def _plot_pca(scores: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for condition, group in scores.groupby("condition"):
        ax.scatter(group.PC1, group.PC2, s=25, alpha=0.8, label=condition, color=PALETTE[condition])
    ax.set_xlabel(f"PC1 ({scores.attrs['explained'][0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({scores.attrs['explained'][1] * 100:.1f}% variance)")
    ax.set_title("PCA of log2 median-ratio-normalized expression")
    ax.legend(title="Condition", frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "pca_by_condition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.scatterplot(data=scores, x="PC1", y="PC2", hue="library_pool", style="chemistry", s=40, ax=ax)
    ax.set_xlabel(f"PC1 ({scores.attrs['explained'][0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({scores.attrs['explained'][1] * 100:.1f}% variance)")
    ax.set_title("PCA colored by library pool")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "pca_by_technical_covariates.png", dpi=180)
    plt.close(fig)


def _plot_qc(counts: pd.DataFrame, sample_metrics: pd.DataFrame, figures: Path) -> None:
    means = counts.mean(axis=1)
    variances = counts.var(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(np.log10(means + 1), np.log10(variances + 1), s=4, alpha=0.2, rasterized=True)
    ax.set_xlabel("log10(mean raw count + 1)")
    ax.set_ylabel("log10(raw-count variance + 1)")
    ax.set_title("Raw-count mean–variance relationship")
    fig.tight_layout()
    fig.savefig(figures / "raw_count_mean_variance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=sample_metrics, x="condition", y="library_size", order=list("ABCD"), ax=axes[0], palette=PALETTE, hue="condition", legend=False)
    axes[0].set_title("Library size by condition")
    axes[0].set_ylabel("Total raw gene counts")
    sns.boxplot(data=sample_metrics, x="condition", y="expressed_genes_ge10", order=list("ABCD"), ax=axes[1], palette=PALETTE, hue="condition", legend=False)
    axes[1].set_title("Genes with count ≥10")
    axes[1].set_ylabel("Number of genes")
    fig.tight_layout()
    fig.savefig(figures / "sample_count_qc.png", dpi=180)
    plt.close(fig)


def _plot_interactions(results: pd.DataFrame, contrasts: pd.DataFrame, figures: Path) -> None:
    significant = results["adjusted_p_value"] < 0.05
    y = -np.log10(results["p_value"].clip(lower=np.nextafter(0, 1)))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(results.loc[~significant, "mean_interaction_log2"], y[~significant], s=7, alpha=0.25, color="#777777", label="FDR ≥ 0.05")
    ax.scatter(results.loc[significant, "mean_interaction_log2"], y[significant], s=10, alpha=0.7, color="#b2182b", label="FDR < 0.05")
    for row in results.head(10).itertuples():
        ax.annotate(row.gene_name if isinstance(row.gene_name, str) else row.gene_id, (row.mean_interaction_log2, -np.log10(max(row.p_value, np.nextafter(0, 1)))), fontsize=7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Mean within-donor interaction (log2 units)")
    ax.set_ylabel("−log10(p-value)")
    ax.set_title("IFN-γ × Salmonella transcriptional interaction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "interaction_volcano.png", dpi=180)
    plt.close(fig)

    top_ids = results.head(min(40, len(results))).gene_id.tolist()
    labels = results.set_index("gene_id").loc[top_ids, "gene_name"].fillna(pd.Series(top_ids, index=top_ids))
    heat = contrasts.loc[top_ids].copy()
    heat.index = [labels.loc[g] if labels.loc[g] else g for g in top_ids]
    grid = sns.clustermap(heat, cmap="vlag", center=0, figsize=(12, 10), xticklabels=False, yticklabels=True, cbar_kws={"label": "D−B−C+A (log2)"})
    grid.fig.suptitle("Top donor-level interaction contrasts", y=1.01)
    grid.fig.savefig(figures / "top_interaction_gene_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(grid.fig)

    top = results.head(min(6, len(results)))
    fig, axes = plt.subplots(len(top), 1, figsize=(9, 1.8 * len(top)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, top.itertuples()):
        vals = contrasts.loc[row.gene_id]
        ax.scatter(range(len(vals)), vals, s=12, alpha=0.65)
        ax.axhline(row.mean_interaction_log2, color="#b2182b", lw=1.5)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(row.gene_name if isinstance(row.gene_name, str) else row.gene_id)
    axes[-1].set_xlabel("Donors (sorted identifier order)")
    fig.suptitle("Per-donor contrasts for top interaction genes")
    fig.tight_layout()
    fig.savefig(figures / "top_interaction_per_donor.png", dpi=180)
    plt.close(fig)


def run(root: Path) -> dict[str, object]:
    published = root / "data/published"
    tables, figures, processed, reports = (root / "results/tables", root / "results/figures", root / "data/processed", root / "results/reports")
    for directory in (tables, figures, processed, reports):
        directory.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(published / "RNA_sample_metadata.txt.gz", sep="\t")
    counts = pd.read_csv(published / "RNA_count_matrix.txt.gz", sep="\t", index_col=0)
    cqn = pd.read_csv(published / "RNA_cqn_matrix.txt.gz", sep="\t", index_col=0)
    genes = pd.read_csv(published / "RNA_gene_metadata.txt.gz", sep="\t")

    normalized, factors = median_of_ratios(counts)
    factors.rename_axis("sample_id").to_csv(tables / "median_ratio_size_factors.csv")
    primary_mask = expressed_filter(counts, 10, 84)
    log_expression = log2_transform(normalized.loc[primary_mask], 0.5)
    log_expression.to_csv(processed / "log2_median_ratio_expression.tsv.gz", sep="\t", compression="gzip")

    sample_metrics = metadata.set_index("sample_id").loc[counts.columns].copy()
    sample_metrics["library_size"] = counts.sum(axis=0)
    sample_metrics["expressed_genes_ge1"] = (counts >= 1).sum(axis=0)
    sample_metrics["expressed_genes_ge10"] = (counts >= 10).sum(axis=0)
    sample_metrics["size_factor"] = factors
    sample_metrics.reset_index().to_csv(tables / "sample_qc_metrics.csv", index=False)
    _plot_qc(counts, sample_metrics.reset_index(), figures)

    variable_ids = log_expression.var(axis=1).nlargest(min(2000, len(log_expression))).index
    pca = PCA(n_components=10, random_state=607)
    scores_array = pca.fit_transform(log_expression.loc[variable_ids].T)
    scores = pd.DataFrame(scores_array, index=log_expression.columns, columns=[f"PC{i}" for i in range(1, 11)])
    scores = scores.join(metadata.set_index("sample_id")[["donor", "condition", "condition_name", "library_pool", "chemistry", "rna_auto"]])
    scores.attrs["explained"] = pca.explained_variance_ratio_
    scores.rename_axis("sample_id").reset_index().to_csv(tables / "pca_scores.csv", index=False)
    pd.DataFrame({"component": range(1, 11), "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative": np.cumsum(pca.explained_variance_ratio_)}).to_csv(tables / "pca_explained_variance.csv", index=False)
    _plot_pca(scores, figures)

    contrasts = interaction_contrasts(log_expression, metadata)
    contrasts.to_csv(processed / "donor_interaction_contrasts.tsv.gz", sep="\t", compression="gzip", index_label="gene_id")
    tested = test_interactions(contrasts)
    results = _annotate(tested, genes)
    results["significant_fdr_0_05"] = results.adjusted_p_value < 0.05
    results.to_csv(tables / "factorial_interaction_results.csv.gz", index=False, compression="gzip")
    _plot_interactions(results, contrasts, figures)

    condition_means = {}
    for condition in "ABCD":
        ids = metadata.loc[metadata.condition == condition, "sample_id"]
        condition_means[condition] = log_expression[ids].mean(axis=1)
    means = pd.DataFrame(condition_means)
    by_name = genes.drop_duplicates("gene_name").set_index("gene_name")
    candidate_rows = []
    indexed = results.set_index("gene_id")
    for name in CANDIDATES:
        if name not in by_name.index:
            candidate_rows.append({"gene_name": name, "annotation_status": "not present"})
            continue
        gene_id = by_name.loc[name, "gene_id"]
        if isinstance(gene_id, pd.Series):
            gene_id = gene_id.iloc[0]
        row = {"gene_name": name, "gene_id": gene_id, "annotation_status": "tested" if gene_id in indexed.index else "filtered_low_expression"}
        if gene_id in means.index:
            row.update({f"mean_log2_{c}": means.loc[gene_id, c] for c in "ABCD"})
        if gene_id in indexed.index:
            row.update(indexed.loc[gene_id, ["mean_interaction_log2", "ci_lower", "ci_upper", "p_value", "adjusted_p_value"]].to_dict())
        candidate_rows.append(row)
    pd.DataFrame(candidate_rows).to_csv(tables / "candidate_gene_summary.csv", index=False)

    robustness_rows = []
    primary_by_gene = tested
    for minimum_count, minimum_samples in [(5, 42), (10, 84), (20, 84)]:
        mask = expressed_filter(counts, minimum_count, minimum_samples)
        expr = log2_transform(normalized.loc[mask], 0.5)
        alt = test_interactions(interaction_contrasts(expr, metadata))
        common = primary_by_gene.index.intersection(alt.index)
        rho = stats.spearmanr(primary_by_gene.loc[common, "mean_interaction_log2"], alt.loc[common, "mean_interaction_log2"]).statistic
        robustness_rows.append({"method": "median_ratio_log2", "minimum_count": minimum_count, "minimum_samples": minimum_samples, "n_genes": int(mask.sum()), "n_fdr_0_05": int((alt.adjusted_p_value < 0.05).sum()), "effect_spearman_vs_primary": rho})
    cqn_masked = cqn.loc[primary_mask]
    cqn_test = test_interactions(interaction_contrasts(cqn_masked, metadata))
    common = primary_by_gene.index.intersection(cqn_test.index)
    robustness_rows.append({"method": "authors_cqn", "minimum_count": 10, "minimum_samples": 84, "n_genes": len(cqn_test), "n_fdr_0_05": int((cqn_test.adjusted_p_value < 0.05).sum()), "effect_spearman_vs_primary": stats.spearmanr(primary_by_gene.loc[common, "mean_interaction_log2"], cqn_test.loc[common, "mean_interaction_log2"]).statistic})
    pd.DataFrame(robustness_rows).to_csv(tables / "interaction_robustness_summary.csv", index=False)

    top_ids = tested.head(min(100, len(tested))).index
    top_contrasts = contrasts.loc[top_ids]
    total = top_contrasts.sum(axis=1)
    loo_means = pd.DataFrame({donor: (total - top_contrasts[donor]) / (top_contrasts.shape[1] - 1) for donor in top_contrasts.columns})
    influence = pd.DataFrame({"primary_mean": top_contrasts.mean(axis=1), "loo_min": loo_means.min(axis=1), "loo_max": loo_means.max(axis=1), "max_abs_loo_change": loo_means.sub(top_contrasts.mean(axis=1), axis=0).abs().max(axis=1)})
    _annotate(influence, genes).to_csv(tables / "leave_one_donor_out_top100.csv", index=False)

    pairwise_rows = []
    for condition, label in [("B", "IFNg_vs_naive"), ("C", "SL1344_vs_naive"), ("D", "IFNg_SL1344_vs_naive")]:
        effect = _paired_effect(log_expression, metadata, condition)
        effect = _annotate(effect, genes)
        effect.to_csv(tables / f"paired_{label}.csv.gz", index=False, compression="gzip")
        pairwise_rows.append({"comparison": label, "n_tested": len(effect), "n_fdr_0_05": int((effect.adjusted_p_value < 0.05).sum()), "n_fdr_0_05_abs_log2_ge_1": int(((effect.adjusted_p_value < 0.05) & (effect.mean_log2_change.abs() >= 1)).sum())})
    pd.DataFrame(pairwise_rows).to_csv(tables / "paired_response_summary.csv", index=False)

    baseline = pd.DataFrame([
        {"finding": "Gene universe after excluding short RNAs and pseudogenes", "published_value": 35033, "our_value": len(genes), "agreement": len(genes) == 35033, "source": "Alasoo et al. methods / published gene metadata"},
        {"finding": "Protein-coding genes in universe", "published_value": 19796, "our_value": int((genes.gene_biotype == "protein_coding").sum()), "agreement": int((genes.gene_biotype == "protein_coding").sum()) == 19796, "source": "Alasoo et al. methods / published gene metadata"},
        {"finding": "Full-cohort profiles", "published_value": 336, "our_value": len(metadata), "agreement": len(metadata) == 336, "source": "published sample metadata"},
        {"finding": "Full-cohort donors", "published_value": 84, "our_value": metadata.donor.nunique(), "agreement": metadata.donor.nunique() == 84, "source": "published sample metadata"},
    ])
    baseline.to_csv(tables / "baseline_reproduction.csv", index=False)

    summary = {
        "n_samples": len(metadata), "n_donors": int(metadata.donor.nunique()), "n_genes_raw": len(counts),
        "n_genes_primary_filter": int(primary_mask.sum()), "n_interaction_fdr_0_05": int((results.adjusted_p_value < 0.05).sum()),
        "median_library_size": float(sample_metrics.library_size.median()), "size_factor_min": float(factors.min()), "size_factor_max": float(factors.max()),
        "pc1_variance_percent": float(100 * pca.explained_variance_ratio_[0]), "pc2_variance_percent": float(100 * pca.explained_variance_ratio_[1]),
    }
    (tables / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

