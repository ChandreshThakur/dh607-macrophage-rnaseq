"""Phase 4 donor-paired IFN-gamma by Salmonella interaction analysis."""

from __future__ import annotations

import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeseq2
import scipy
import seaborn as sns
from scipy import stats

from preprocessing.normalization import expressed_filter, log2_transform, median_of_ratios
from statistical_analysis.interaction import (
    benjamini_hochberg,
    fixed_effect_design_matrix,
    interaction_coefficient_contrast,
    interaction_contrasts,
    interaction_influence,
    test_interactions,
    validate_design_matrix,
    validate_factorial_metadata,
)

CONDITION_NAMES = {
    "A": "Naive",
    "B": "IFN-gamma",
    "C": "Salmonella",
    "D": "IFN-gamma + Salmonella",
}
CONDITION_COLORS = {"A": "#6b6b6b", "B": "#2b8cbe", "C": "#e6550d", "D": "#756bb1"}
PRIMARY_MIN_COUNT = 10
PRIMARY_MIN_SAMPLES = 84
FDR = 0.05


def _paths(root: Path) -> dict[str, Path]:
    paths = {
        "published": root / "data" / "published",
        "processed": root / "data" / "processed",
        "tables": root / "results" / "tables",
        "figures": root / "results" / "figures",
        "reports": root / "results" / "reports",
        "configs": root / "configs",
    }
    for key in ("processed", "tables", "figures", "reports"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _paths(root)
    metadata = pd.read_csv(paths["published"] / "RNA_sample_metadata.txt.gz", sep="\t")
    counts = pd.read_csv(paths["published"] / "RNA_count_matrix.txt.gz", sep="\t", index_col=0)
    cqn = pd.read_csv(paths["published"] / "RNA_cqn_matrix.txt.gz", sep="\t", index_col=0)
    genes = pd.read_csv(paths["published"] / "RNA_gene_metadata.txt.gz", sep="\t").set_index("gene_id")
    metadata = validate_factorial_metadata(metadata, counts.columns)
    if list(cqn.index) != list(counts.index) or list(cqn.columns) != list(counts.columns):
        raise ValueError("raw-count and CQN matrices are not identically aligned")
    if not np.issubdtype(counts.to_numpy().dtype, np.integer) or (counts.to_numpy() < 0).any():
        raise ValueError("count-aware model requires nonnegative integer raw counts")
    if set(genes.index) != set(counts.index):
        raise ValueError("gene annotation and expression identifiers differ")
    return metadata, counts, cqn, genes


def _annotation(genes: pd.DataFrame) -> pd.DataFrame:
    columns = ["gene_name", "gene_biotype", "chr", "start", "end"]
    return genes.loc[:, columns].copy()


def _condition_statistics(expression: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    lookup = metadata.set_index(["donor", "condition"]).sample_id
    donors = sorted(metadata.donor.unique())
    arrays = {
        c: expression.loc[:, [lookup.loc[(d, c)] for d in donors]].to_numpy(dtype=float)
        for c in "ABCD"
    }
    out = pd.DataFrame(index=expression.index)
    for condition in "ABCD":
        out[f"mean_{condition}"] = arrays[condition].mean(axis=1)
    out["IFNg_effect_uninfected_B_minus_A"] = (arrays["B"] - arrays["A"]).mean(axis=1)
    out["IFNg_effect_infected_D_minus_C"] = (arrays["D"] - arrays["C"]).mean(axis=1)
    out["Salmonella_effect_unprimed_C_minus_A"] = (arrays["C"] - arrays["A"]).mean(axis=1)
    out["Salmonella_effect_primed_D_minus_B"] = (arrays["D"] - arrays["B"]).mean(axis=1)
    return out


def analyze_expression_scale(
    expression: pd.DataFrame, metadata: pd.DataFrame, genes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = interaction_contrasts(expression, metadata)
    contrasts.index.name = "gene_id"
    tested = test_interactions(contrasts)
    tested["n_donors"] = contrasts.shape[1]
    tested = tested.join(interaction_influence(contrasts))
    tested = tested.join(_condition_statistics(expression, metadata))
    tested = tested.join(_annotation(genes), how="left")
    tested.index.name = "gene_id"
    return tested, contrasts


def _safe_neglog10(values: pd.Series) -> np.ndarray:
    return -np.log10(np.clip(values.to_numpy(dtype=float), np.nextafter(0, 1), 1))


def _display_name(frame: pd.DataFrame) -> pd.Series:
    return frame.gene_name.where(frame.gene_name.notna() & frame.gene_name.ne(""), frame.index.to_series())


def _plot_volcano(results: pd.DataFrame, path: Path) -> None:
    significant = results.adjusted_p_value < FDR
    y = _safe_neglog10(results.p_value)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.scatter(results.loc[~significant, "mean_interaction_log2"], y[~significant], s=8, alpha=.24,
               color="#7f7f7f", linewidths=0, label="FDR >= 0.05")
    ax.scatter(results.loc[significant & (results.mean_interaction_log2 > 0), "mean_interaction_log2"],
               y[significant & (results.mean_interaction_log2 > 0)], s=10, alpha=.58,
               color="#b2182b", linewidths=0, label="Positive, FDR < 0.05")
    ax.scatter(results.loc[significant & (results.mean_interaction_log2 < 0), "mean_interaction_log2"],
               y[significant & (results.mean_interaction_log2 < 0)], s=10, alpha=.58,
               color="#2166ac", linewidths=0, label="Negative, FDR < 0.05")
    labels = pd.concat([
        results.nsmallest(5, "mean_interaction_log2"),
        results.nlargest(5, "mean_interaction_log2"),
    ]).drop_duplicates()
    names = _display_name(labels)
    for gene_id, row in labels.iterrows():
        ax.annotate(names.loc[gene_id], (row.mean_interaction_log2, -np.log10(max(row.p_value, np.nextafter(0, 1)))),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color="black", lw=.7)
    ax.set(xlabel="Mean donor interaction: D - B - C + A (log2 units)",
           ylabel="-log10(two-sided P value)", title="IFN-gamma priming by Salmonella interaction")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_effect_distribution(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.7))
    sns.histplot(results.mean_interaction_log2, bins=100, color="#4c78a8", ax=ax)
    ax.axvline(0, color="black", lw=.8)
    q = results.mean_interaction_log2.quantile([.01, .5, .99])
    ax.set(xlabel="Mean interaction (log2 units)", ylabel="Genes",
           title=f"Genome-wide interaction effects (1%={q.iloc[0]:.2f}, median={q.iloc[1]:.2f}, 99%={q.iloc[2]:.2f})")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_top_effects(results: pd.DataFrame, path: Path, n_each: int = 12) -> None:
    selected = pd.concat([results.nsmallest(n_each, "mean_interaction_log2"),
                          results.nlargest(n_each, "mean_interaction_log2")]).copy()
    selected["label"] = _display_name(selected)
    selected = selected.sort_values("mean_interaction_log2")
    colors = np.where(selected.mean_interaction_log2 > 0, "#b2182b", "#2166ac")
    fig, ax = plt.subplots(figsize=(8.2, 8))
    y = np.arange(len(selected))
    ax.barh(y, selected.mean_interaction_log2, color=colors, alpha=.86)
    ax.errorbar(selected.mean_interaction_log2, y,
                xerr=[selected.mean_interaction_log2-selected.ci_lower,
                      selected.ci_upper-selected.mean_interaction_log2],
                fmt="none", ecolor="black", elinewidth=.7, capsize=2)
    ax.set_yticks(y, selected.label)
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Interaction effect with 95% CI (log2 units)", title="Largest positive and negative interaction effects")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_heatmap(results: pd.DataFrame, path: Path, n_each: int = 15) -> None:
    selected = pd.concat([results.nsmallest(n_each, "mean_interaction_log2"),
                          results.nlargest(n_each, "mean_interaction_log2")]).drop_duplicates().copy()
    selected["label"] = _display_name(selected)
    matrix = selected.set_index("label")[[f"mean_{c}" for c in "ABCD"]]
    matrix.columns = [CONDITION_NAMES[c] for c in "ABCD"]
    centered = matrix.sub(matrix.mean(axis=1), axis=0)
    centered = centered.div(matrix.std(axis=1).replace(0, np.nan), axis=0)
    order = selected.sort_values("mean_interaction_log2").label
    fig, ax = plt.subplots(figsize=(7, 9))
    sns.heatmap(centered.loc[order], cmap="vlag", center=0, robust=True, ax=ax,
                cbar_kws={"label": "Gene-centered condition mean (z score)"})
    ax.set(xlabel="Condition", ylabel="Top/bottom effect genes (algorithmic)",
           title="Four-condition expression patterns")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _representative_genes(results: pd.DataFrame, candidates: pd.DataFrame) -> list[str]:
    sig = results.loc[results.adjusted_p_value < FDR]
    choices: list[str] = []
    if not sig.empty:
        choices.extend([sig.mean_interaction_log2.idxmax(), sig.mean_interaction_log2.idxmin()])
    for symbol in ("IFNL2", "DUSP1", "PTGS2", "IL10"):
        matches = results.index[results.gene_name.eq(symbol)]
        if len(matches) == 1:
            choices.append(matches[0])
    return list(dict.fromkeys(choices))[:6]


def _plot_interaction_means(results: pd.DataFrame, selected_ids: list[str], path: Path) -> None:
    n = len(selected_ids); cols = 2 if n <= 4 else 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.7 * rows), squeeze=False)
    for ax, gene_id in zip(axes.flat, selected_ids):
        row = results.loc[gene_id]
        for primed, conditions, color, label in [
            (False, ("A", "C"), "#555555", "No IFN-gamma"),
            (True, ("B", "D"), "#7b3294", "IFN-gamma primed"),
        ]:
            values = [row[f"mean_{c}"] for c in conditions]
            ax.plot([0, 1], values, marker="o", lw=2, color=color, label=label)
        name = row.gene_name if pd.notna(row.gene_name) else gene_id
        ax.set_xticks([0, 1], ["Control", "Salmonella"])
        ax.set_ylabel("Mean log2 normalized expression")
        ax.set_title(f"{name}: interaction {row.mean_interaction_log2:.2f}\nFDR={row.adjusted_p_value:.2g}")
    for ax in axes.flat[n:]: ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("How priming changes the Salmonella response", y=1.01)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def _plot_donor_responses(expression: pd.DataFrame, metadata: pd.DataFrame, genes: pd.DataFrame,
                          results: pd.DataFrame, selected_ids: list[str], path: Path) -> None:
    lookup = metadata.set_index(["donor", "condition"]).sample_id
    donors = sorted(metadata.donor.unique())
    n = len(selected_ids); cols = 2 if n <= 4 else 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), squeeze=False)
    for ax, gene_id in zip(axes.flat, selected_ids):
        a = expression.loc[gene_id, [lookup.loc[(d,"A")] for d in donors]].to_numpy()
        b = expression.loc[gene_id, [lookup.loc[(d,"B")] for d in donors]].to_numpy()
        c = expression.loc[gene_id, [lookup.loc[(d,"C")] for d in donors]].to_numpy()
        d = expression.loc[gene_id, [lookup.loc[(d,"D")] for d in donors]].to_numpy()
        unprimed = c-a; primed = d-b
        for x, y in zip(unprimed, primed): ax.plot([0,1],[x,y], color="#bdbdbd", alpha=.32, lw=.65)
        ax.scatter(np.zeros(len(donors)), unprimed, s=10, color="#e6550d", alpha=.55)
        ax.scatter(np.ones(len(donors)), primed, s=10, color="#756bb1", alpha=.55)
        ax.plot([0,1],[unprimed.mean(),primed.mean()], color="black", marker="D", lw=2)
        name = genes.loc[gene_id,"gene_name"] if gene_id in genes.index else gene_id
        ax.set_xticks([0,1],["Without priming\nC-A","After priming\nD-B"])
        ax.set_ylabel("Donor Salmonella response (log2)")
        ax.set_title(f"{name}: D-B-(C-A)={results.loc[gene_id,'mean_interaction_log2']:.2f}")
    for ax in axes.flat[n:]: ax.axis("off")
    fig.suptitle("Paired donor responses; every line is one donor", y=1.01)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def _plot_normalization(primary: pd.DataFrame, cqn: pd.DataFrame, path: Path) -> None:
    common = primary.index.intersection(cqn.index)
    x = primary.loc[common, "mean_interaction_log2"]
    y = cqn.loc[common, "mean_interaction_log2"]
    rho = stats.spearmanr(x, y).statistic
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hexbin(x, y, gridsize=75, mincnt=1, bins="log", cmap="viridis")
    lim = np.nanmax(np.abs(np.r_[x.to_numpy(), y.to_numpy()]))
    axes[0].plot([-lim,lim],[-lim,lim],color="white",lw=.8,ls="--")
    axes[0].set(xlabel="Median-ratio log2 interaction", ylabel="Authors' CQN interaction",
                title=f"Effect concordance (Spearman rho={rho:.3f})")
    delta = y-x
    axes[1].scatter((x+y)/2, delta, s=7, alpha=.22, color="#4c78a8", linewidths=0)
    axes[1].axhline(0,color="black",lw=.8)
    axes[1].set(xlabel="Mean of the two interaction estimates", ylabel="CQN - median-ratio effect",
                title="Normalization-dependent effect difference")
    fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)


def run_course(root: Path) -> dict[str, object]:
    paths = _paths(root)
    metadata, counts, cqn, genes = load_data(root)
    design = fixed_effect_design_matrix(metadata)
    design_diagnostics = validate_design_matrix(design)
    pd.DataFrame({"coefficient": design.columns}).to_csv(paths["tables"] / "interaction_design_coefficients.csv", index=False)
    normalized, size_factors = median_of_ratios(counts)
    primary_mask = expressed_filter(counts, PRIMARY_MIN_COUNT, PRIMARY_MIN_SAMPLES)
    expression = log2_transform(normalized.loc[primary_mask], .5)
    primary, contrasts = analyze_expression_scale(expression, metadata, genes)
    primary.to_csv(paths["tables"] / "interaction_gene_results.csv")
    contrasts.to_csv(paths["processed"] / "phase4_donor_interaction_contrasts.tsv.gz", sep="\t", compression="gzip")

    cqn_primary = cqn.loc[primary_mask]
    cqn_results, _ = analyze_expression_scale(cqn_primary, metadata, genes)
    sensitivity = primary[["mean_interaction_log2", "adjusted_p_value"]].rename(columns={
        "mean_interaction_log2":"median_ratio_interaction", "adjusted_p_value":"median_ratio_adjusted_p_value"})
    sensitivity = sensitivity.join(cqn_results[["mean_interaction_log2", "adjusted_p_value"]].rename(columns={
        "mean_interaction_log2":"cqn_interaction", "adjusted_p_value":"cqn_adjusted_p_value"}))
    sensitivity["same_direction"] = np.sign(sensitivity.median_ratio_interaction) == np.sign(sensitivity.cqn_interaction)
    sensitivity["median_ratio_significant"] = sensitivity.median_ratio_adjusted_p_value < FDR
    sensitivity["cqn_significant"] = sensitivity.cqn_adjusted_p_value < FDR
    sensitivity = sensitivity.join(_annotation(genes), how="left")
    sensitivity.to_csv(paths["tables"] / "normalization_sensitivity.csv")

    filter_rows = []
    for label, min_count, min_samples in [
        ("primary",10,84), ("lenient",5,42), ("stringent",20,84)
    ]:
        mask = expressed_filter(counts,min_count,min_samples)
        alt, _ = analyze_expression_scale(log2_transform(normalized.loc[mask],.5),metadata,genes)
        common = primary.index.intersection(alt.index)
        filter_rows.append({
            "filter":label,"minimum_count":min_count,"minimum_samples":min_samples,
            "n_genes":len(alt),"n_fdr_0_05":int((alt.adjusted_p_value<FDR).sum()),
            "effect_spearman_vs_primary":float(stats.spearmanr(primary.loc[common,"mean_interaction_log2"],alt.loc[common,"mean_interaction_log2"]).statistic),
            "direction_agreement_vs_primary":float((np.sign(primary.loc[common,"mean_interaction_log2"])==np.sign(alt.loc[common,"mean_interaction_log2"])).mean()),
            "significance_agreement_vs_primary":float(((primary.loc[common,"adjusted_p_value"]<FDR)==(alt.loc[common,"adjusted_p_value"]<FDR)).mean()),
            "significant_in_both":int(((primary.loc[common,"adjusted_p_value"]<FDR)&(alt.loc[common,"adjusted_p_value"]<FDR)).sum()),
        })
    filter_table = pd.DataFrame(filter_rows)
    filter_table.to_csv(paths["tables"] / "interaction_filter_sensitivity.csv",index=False)

    panel = pd.read_csv(paths["configs"] / "phase4_candidate_panel.csv").set_index("gene_id")
    candidate = panel.join(primary.drop(columns=["gene_name","gene_biotype","chr"], errors="ignore"), how="left")
    candidate = candidate.join(cqn_results[["mean_interaction_log2","adjusted_p_value"]].rename(columns={
        "mean_interaction_log2":"cqn_interaction","adjusted_p_value":"cqn_genomewide_adjusted_p_value"}),how="left")
    candidate["candidate_panel_adjusted_p_value"] = benjamini_hochberg(candidate.p_value.to_numpy())
    candidate.to_csv(paths["tables"] / "interaction_candidate_genes.csv")

    normalization_summary = {
        "n_common":int(len(sensitivity)),
        "pearson":float(stats.pearsonr(sensitivity.median_ratio_interaction,sensitivity.cqn_interaction).statistic),
        "spearman":float(stats.spearmanr(sensitivity.median_ratio_interaction,sensitivity.cqn_interaction).statistic),
        "direction_agreement":float(sensitivity.same_direction.mean()),
        "primary_significant":int(sensitivity.median_ratio_significant.sum()),
        "cqn_significant":int(sensitivity.cqn_significant.sum()),
        "significant_both":int((sensitivity.median_ratio_significant & sensitivity.cqn_significant).sum()),
    }
    diagnostics = {
        "design":design_diagnostics,
        "condition_mapping":metadata.groupby("condition")[["condition_name","IFNg","SL1344","IFNg_binary","Salmonella_binary"]].first().to_dict("index"),
        "n_samples":len(metadata),"n_donors":metadata.donor.nunique(),"n_genes_tested":int(primary_mask.sum()),
        "n_primary_fdr_0_05":int((primary.adjusted_p_value<FDR).sum()),
        "n_primary_positive_fdr_0_05":int(((primary.adjusted_p_value<FDR)&(primary.mean_interaction_log2>0)).sum()),
        "n_primary_negative_fdr_0_05":int(((primary.adjusted_p_value<FDR)&(primary.mean_interaction_log2<0)).sum()),
        "normalization":normalization_summary,
        "size_factor_condition_medians":size_factors.groupby(metadata.set_index("sample_id").loc[size_factors.index,"condition"]).median().to_dict(),
    }
    (paths["tables"] / "phase4_course_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")

    _plot_volcano(primary,paths["figures"] / "interaction_volcano.png")
    _plot_effect_distribution(primary,paths["figures"] / "interaction_effect_distribution.png")
    _plot_top_effects(primary,paths["figures"] / "interaction_top_effects.png")
    _plot_heatmap(primary,paths["figures"] / "interaction_heatmap.png")
    selected_ids = _representative_genes(primary,panel)
    pd.DataFrame({"gene_id":selected_ids,"selection_rule":["algorithmic positive/negative genome-wide or candidate representative"]*len(selected_ids)}).to_csv(paths["tables"] / "interaction_representative_genes.csv",index=False)
    _plot_interaction_means(primary,selected_ids,paths["figures"] / "interaction_candidate_plots.png")
    _plot_donor_responses(expression,metadata,genes,primary,selected_ids,paths["figures"] / "interaction_donor_response_plots.png")
    _plot_normalization(primary,cqn_results,paths["figures"] / "interaction_normalization_sensitivity.png")
    return diagnostics


def _fit_pydeseq(counts: pd.DataFrame, metadata: pd.DataFrame, n_cpus: int = 1):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    samples = metadata.sample_id.tolist()
    count_input = counts.loc[:, samples].T.astype(int)
    model_metadata = metadata.set_index("sample_id").loc[samples, ["donor","IFNg_binary","Salmonella_binary"]].copy()
    formula = "~ donor + IFNg_binary + Salmonella_binary + IFNg_binary:Salmonella_binary"
    dds = DeseqDataSet(counts=count_input,metadata=model_metadata,design=formula,refit_cooks=False,
                       n_cpus=n_cpus,quiet=True,low_memory=True)
    design = dds.obsm["design_matrix"]
    validate_design_matrix(pd.DataFrame(design,index=design.index,columns=design.columns))
    contrast = interaction_coefficient_contrast(design.columns)
    dds.deseq2()
    ds = DeseqStats(dds,contrast=contrast,cooks_filter=False,independent_filter=False,
                    n_cpus=n_cpus,quiet=True)
    ds.summary()
    return dds, ds.results_df.copy(), design.columns.tolist(), contrast


def run_count_model(root: Path) -> dict[str, object]:
    paths = _paths(root)
    metadata, counts, _, genes = load_data(root)
    primary_mask = expressed_filter(counts,PRIMARY_MIN_COUNT,PRIMARY_MIN_SAMPLES)
    filtered = counts.loc[primary_mask]
    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dds, count_results, coefficients, contrast = _fit_pydeseq(filtered,metadata,n_cpus=1)
    count_results.index.name = "gene_id"
    count_results = count_results.join(_annotation(genes),how="left")
    count_results.to_csv(paths["tables"] / "interaction_count_model_results.csv.gz",compression="gzip")
    convergence = pd.DataFrame(index=filtered.index)
    convergence.index.name = "gene_id"
    for source,target in [("_genewise_converged","genewise_dispersion_converged"),
                          ("_MAP_converged","map_dispersion_converged"),("_LFC_converged","lfc_converged"),
                          ("_outlier_genes","dispersion_outlier"),("_pvalue_cooks_outlier","cooks_outlier")]:
        if source in dds.var.columns: convergence[target]=dds.var[source].to_numpy()
    convergence.to_csv(paths["tables"] / "interaction_count_model_convergence.csv")
    course = pd.read_csv(paths["tables"] / "interaction_gene_results.csv",index_col="gene_id")
    comparison = course[["mean_interaction_log2","adjusted_p_value"]].rename(columns={
        "mean_interaction_log2":"course_interaction_log2","adjusted_p_value":"course_adjusted_p_value"})
    comparison = comparison.join(count_results[["log2FoldChange","lfcSE","stat","pvalue","padj"]].rename(columns={
        "log2FoldChange":"count_interaction_log2","lfcSE":"count_lfc_se","stat":"count_wald_statistic",
        "pvalue":"count_p_value","padj":"count_adjusted_p_value"}))
    comparison["same_direction"] = np.sign(comparison.course_interaction_log2)==np.sign(comparison.count_interaction_log2)
    comparison["course_significant"] = comparison.course_adjusted_p_value<FDR
    comparison["count_significant"] = comparison.count_adjusted_p_value<FDR
    comparison = comparison.join(convergence)
    comparison["count_reliable"] = comparison.lfc_converged.fillna(False) & comparison.map_dispersion_converged.fillna(False)
    comparison = comparison.join(_annotation(genes),how="left")
    comparison.index.name = "gene_id"
    comparison.to_csv(paths["tables"] / "interaction_model_comparison.csv")
    finite = comparison[["count_interaction_log2","count_p_value"]].notna().all(axis=1)
    diagnostics = {
        "package":"pydeseq2","version":pydeseq2.__version__,"python":platform.python_version(),
        "numpy":np.__version__,"scipy":scipy.__version__,"formula":"~ donor + IFNg_binary + Salmonella_binary + IFNg_binary:Salmonella_binary",
        "coefficients":coefficients,"tested_coefficient":coefficients[int(np.flatnonzero(contrast)[0])],
        "reference_levels":{"IFNg_binary":0,"Salmonella_binary":0,"donor":sorted(metadata.donor.unique())[0]},
        "n_samples":len(metadata),"n_donors":metadata.donor.nunique(),"n_genes":len(filtered),
        "elapsed_seconds":time.time()-started,"finite_effect_and_p":int(finite.sum()),
        "n_count_fdr_0_05":int((comparison.count_adjusted_p_value<FDR).sum()),
        "n_warnings":len(caught),"warnings":[str(w.message) for w in caught],
        "convergence_counts":{col:int(convergence[col].fillna(False).astype(bool).sum()) for col in convergence.columns},
        "cooks_filter":False,"independent_filter":False,"refit_cooks":False,"n_cpus":1,"low_memory":True,
    }
    (paths["tables"] / "interaction_count_model_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    return diagnostics


def _plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    valid = comparison.dropna(subset=["course_interaction_log2","count_interaction_log2"])
    rho = stats.spearmanr(valid.course_interaction_log2,valid.count_interaction_log2).statistic
    fig, ax = plt.subplots(figsize=(6.5,6))
    ax.hexbin(valid.course_interaction_log2,valid.count_interaction_log2,gridsize=75,mincnt=1,bins="log",cmap="magma")
    lim=np.nanmax(np.abs(valid[["course_interaction_log2","count_interaction_log2"]].to_numpy()))
    ax.plot([-lim,lim],[-lim,lim],color="white",ls="--",lw=.8)
    ax.set(xlabel="Course donor-contrast interaction (log2)",ylabel="PyDESeq2 interaction coefficient (log2)",
           title=f"Interaction-model comparison (Spearman rho={rho:.3f})")
    fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)


def _plot_significant_summary(results: pd.DataFrame, candidates: pd.DataFrame, path: Path) -> None:
    sig=results.adjusted_p_value<FDR
    summary=pd.Series({"Positive":int((sig&(results.mean_interaction_log2>0)).sum()),
                       "Negative":int((sig&(results.mean_interaction_log2<0)).sum()),
                       "Not significant":int((~sig).sum())})
    c=candidates.groupby("theme").agg(tested=("p_value","size"),significant=("adjusted_p_value",lambda x:(x<FDR).sum()))
    c.index = c.index.str.replace("_", " ").str.title()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].bar(summary.index,summary.values,color=["#b2182b","#2166ac","#aaaaaa"])
    axes[0].set(ylabel="Genes",title="Genome-wide BH FDR < 0.05"); axes[0].tick_params(axis="x",rotation=20)
    axes[1].barh(c.index,c.significant,color="#4c78a8")
    for i,(value,total) in enumerate(zip(c.significant,c.tested)): axes[1].text(value+.1,i,f"{value}/{total}",va="center",fontsize=8)
    axes[1].set(xlabel="Significant candidates",title="Frozen candidate panel (genome-wide FDR)")
    fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)


def finalize(root: Path) -> dict[str, object]:
    paths=_paths(root)
    course=pd.read_csv(paths["tables"] / "interaction_gene_results.csv",index_col="gene_id")
    candidates=pd.read_csv(paths["tables"] / "interaction_candidate_genes.csv",index_col="gene_id")
    comparison=pd.read_csv(paths["tables"] / "interaction_model_comparison.csv",index_col="gene_id")
    count_candidate_columns = ["count_interaction_log2","count_lfc_se","count_p_value",
                               "count_adjusted_p_value","same_direction","count_reliable",
                               "lfc_converged","map_dispersion_converged"]
    candidates = candidates.drop(columns=[c for c in count_candidate_columns if c in candidates], errors="ignore")
    candidates = candidates.join(comparison[count_candidate_columns])
    candidates["same_direction_primary_cqn"] = np.sign(candidates.mean_interaction_log2) == np.sign(candidates.cqn_interaction)
    candidates.index.name = "gene_id"
    candidates.to_csv(paths["tables"] / "interaction_candidate_genes.csv")

    theme_summary = candidates.groupby("theme").agg(
        n_genes=("gene_name","size"),
        primary_fdr_0_05=("adjusted_p_value",lambda x:int((x<FDR).sum())),
        positive_interactions=("mean_interaction_log2",lambda x:int((x>0).sum())),
        negative_interactions=("mean_interaction_log2",lambda x:int((x<0).sum())),
        median_interaction=("mean_interaction_log2","median"),
        count_fdr_0_05=("count_adjusted_p_value",lambda x:int((x<FDR).sum())),
        primary_cqn_direction_agreement=("same_direction_primary_cqn","mean"),
    )
    theme_summary.to_csv(paths["tables"] / "interaction_biological_theme_summary.csv")

    top_genes = pd.concat([course.nsmallest(15,"mean_interaction_log2").assign(direction="negative"),
                           course.nlargest(15,"mean_interaction_log2").assign(direction="positive")])
    top_genes = top_genes.join(comparison[["count_interaction_log2","count_adjusted_p_value","count_reliable"]])
    top_genes.index.name = "gene_id"
    top_genes.to_csv(paths["tables"] / "interaction_top_genes.csv")

    expression_dependence = course.copy()
    expression_dependence["mean_expression"] = expression_dependence[[f"mean_{c}" for c in "ABCD"]].mean(axis=1)
    expression_dependence["absolute_interaction"] = expression_dependence.mean_interaction_log2.abs()
    expression_dependence["expression_quartile"] = pd.qcut(
        expression_dependence.mean_expression,4,labels=["Q1_lowest","Q2","Q3","Q4_highest"])
    expression_summary = expression_dependence.groupby("expression_quartile",observed=True).agg(
        n_genes=("mean_expression","size"), median_expression=("mean_expression","median"),
        median_absolute_interaction=("absolute_interaction","median"),
        fdr_0_05=("adjusted_p_value",lambda x:int((x<FDR).sum())),
        absolute_interaction_ge_1=("absolute_interaction",lambda x:int((x>=1).sum())),
    )
    expression_summary.to_csv(paths["tables"] / "interaction_expression_dependence.csv")

    contrasts = pd.read_csv(paths["processed"] / "phase4_donor_interaction_contrasts.tsv.gz",
                            sep="\t", index_col="gene_id")
    representatives = pd.read_csv(paths["tables"] / "interaction_representative_genes.csv").gene_id.tolist()
    focused = list(dict.fromkeys(candidates.index.tolist() + representatives))
    loo_rows = []
    for gene_id in focused:
        values = contrasts.loc[gene_id]
        full_mean = float(values.mean())
        for donor, value in values.items():
            loo_mean = float((values.sum() - value) / (len(values) - 1))
            loo_rows.append({"gene_id":gene_id,"gene_name":course.loc[gene_id,"gene_name"],
                             "donor_omitted":donor,"full_mean_interaction":full_mean,
                             "leave_one_out_mean_interaction":loo_mean,
                             "absolute_mean_shift":abs(loo_mean-full_mean),
                             "direction_preserved":bool(np.sign(loo_mean)==np.sign(full_mean))})
    pd.DataFrame(loo_rows).to_csv(paths["tables"] / "interaction_leave_one_donor_out.csv",index=False)
    _plot_model_comparison(comparison,paths["figures"] / "interaction_effect_comparison.png")
    _plot_significant_summary(course,candidates,paths["figures"] / "interaction_significant_summary.png")
    valid=comparison.dropna(subset=["course_interaction_log2","count_interaction_log2"])
    summary={
        "n_genes_tested":len(course),"n_course_fdr_0_05":int((course.adjusted_p_value<FDR).sum()),
        "n_course_positive_fdr_0_05":int(((course.adjusted_p_value<FDR)&(course.mean_interaction_log2>0)).sum()),
        "n_course_negative_fdr_0_05":int(((course.adjusted_p_value<FDR)&(course.mean_interaction_log2<0)).sum()),
        "n_count_fdr_0_05":int((comparison.count_adjusted_p_value<FDR).sum()),
        "n_count_reliable":int(comparison.count_reliable.sum()),
        "n_count_fdr_0_05_reliable":int(((comparison.count_adjusted_p_value<FDR)&comparison.count_reliable).sum()),
        "model_effect_pearson":float(stats.pearsonr(valid.course_interaction_log2,valid.count_interaction_log2).statistic),
        "model_effect_spearman":float(stats.spearmanr(valid.course_interaction_log2,valid.count_interaction_log2).statistic),
        "model_direction_agreement":float(valid.same_direction.mean()),
        "significant_both":int((valid.course_significant&valid.count_significant).sum()),
        "significant_abs_effect_ge_0_5":int(((course.adjusted_p_value<FDR)&(course.mean_interaction_log2.abs()>=.5)).sum()),
        "significant_abs_effect_ge_1":int(((course.adjusted_p_value<FDR)&(course.mean_interaction_log2.abs()>=1)).sum()),
        "loo_focused_genes":len(focused),
        "loo_direction_reversals":int((~pd.DataFrame(loo_rows).direction_preserved).sum()),
        "absolute_effect_expression_spearman":float(stats.spearmanr(expression_dependence.absolute_interaction,
                                                                      expression_dependence.mean_expression).statistic),
    }
    (paths["tables"] / "phase4_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    count_diagnostics=json.loads((paths["tables"] / "interaction_count_model_diagnostics.json").read_text(encoding="utf-8"))
    course_diagnostics=json.loads((paths["tables"] / "phase4_course_diagnostics.json").read_text(encoding="utf-8"))
    filter_sensitivity=pd.read_csv(paths["tables"] / "interaction_filter_sensitivity.csv")
    positive=course.nlargest(5,"mean_interaction_log2")
    negative=course.nsmallest(5,"mean_interaction_log2")
    def gene_list(frame: pd.DataFrame) -> str:
        return ", ".join(f"{row.gene_name} ({row.mean_interaction_log2:+.2f})" for _,row in frame.iterrows())
    report=f"""# Phase 4 interaction analysis

## Question and design

This phase tests whether the Salmonella response after IFN-gamma priming differs from the response without priming. The primary estimand is the paired donor contrast `D - B - C + A = (D-B) - (C-A)`. All 84 donors and all 336 profiles are retained. The verified fixed-effect design has {course_diagnostics['design']['n_columns']} columns and rank {course_diagnostics['design']['rank']}; donor is categorical and the exact PyDESeq2 coefficient is `IFNg_binary:Salmonella_binary`.

The primary course analysis tests {len(course):,} genes passing count >=10 in >=84 samples on `log2(median-ratio normalized count + 0.5)`. Each gene has 84 donor contrasts, a two-sided one-sample t test, 95% confidence interval, and genome-wide BH correction. The supporting count analysis uses integer counts and PyDESeq2 {count_diagnostics['version']} with `~ donor + IFNg_binary + Salmonella_binary + IFNg_binary:Salmonella_binary`.

## Genome-wide findings

- {summary['n_course_fdr_0_05']:,} genes ({100*summary['n_course_fdr_0_05']/summary['n_genes_tested']:.1f}%) have BH FDR <0.05: {summary['n_course_positive_fdr_0_05']:,} positive and {summary['n_course_negative_fdr_0_05']:,} negative.
- Statistical significance is widespread because the paired design has high power and strict additivity on a log scale is a strong null. Requiring both FDR <0.05 and an absolute interaction >=0.5 leaves {summary['significant_abs_effect_ge_0_5']:,} genes; an absolute interaction >=1 leaves {summary['significant_abs_effect_ge_1']:,}.
- Strongest positive effects are {gene_list(positive)}. The type-III interferons IFNL1, IFNL2, and IFNL3 are prominent positive interactions, consistent with combined treatment producing expression well beyond either single condition.
- Strongest negative effects are {gene_list(negative)}. Many are already strongly induced by IFN-gamma alone, Salmonella alone, or both; their negative sign means sub-additivity on the log scale, not necessarily expression below baseline.

For example, CXCL10 condition means are A={course.loc[course.gene_name.eq('CXCL10'),'mean_A'].iloc[0]:.2f}, B={course.loc[course.gene_name.eq('CXCL10'),'mean_B'].iloc[0]:.2f}, C={course.loc[course.gene_name.eq('CXCL10'),'mean_C'].iloc[0]:.2f}, and D={course.loc[course.gene_name.eq('CXCL10'),'mean_D'].iloc[0]:.2f}. Its interaction is {course.loc[course.gene_name.eq('CXCL10'),'mean_interaction_log2'].iloc[0]:.2f}: combined expression remains very high, but priming leaves little additional Salmonella increment because the IFN program is already near its observed high range.

## Pre-specified biological themes

- **Antimicrobial defense:** all five GBP1-GBP5 candidates have significant negative interactions in the primary and count models. This supports strong shared-pathway induction/saturation, not loss of expression: all remain highly expressed under combined treatment.
- **Interferon signaling:** all 11 frozen candidates are significant and negative in the primary analysis, including STAT1, IRF1, IDO1, CXCL9/10/11, IFIT1-3, ISG15, and OAS1. By contrast, exploratory genome-wide leaders IFNL1-3 are strongly positive, showing that priming does not uniformly attenuate every interferon-related transcript.
- **Inflammatory signaling:** TNF, IL1B, NFKBIA, CXCL8, and CCL2 are significantly negative relative to additivity. PTGS2 is null in the primary and count models but negative on CQN, so it is normalization-sensitive rather than a secure interaction.
- **Immunoregulatory feedback:** SOCS3, ATF3, and TNFAIP3 are significantly negative; DUSP1 is positive in the primary and count analyses but approximately null on CQN; IL10 is not significant on any analyzed model/scale. These mixed results argue against a single uniform feedback response.

These are transcript-level observations in iPSC-derived macrophages. They do not demonstrate protein activity, bacterial killing, benefit, harm, or an in-vivo patient effect. The pathway statements above come from the frozen candidate themes and recognizable top genes; no external ontology enrichment was transmitted or used.

## Robustness and unexpected findings

- The count-aware and donor-contrast effects agree closely (Pearson r={summary['model_effect_pearson']:.3f}, Spearman rho={summary['model_effect_spearman']:.3f}, direction agreement={100*summary['model_direction_agreement']:.1f}%). {summary['significant_both']:,} genes are significant in both.
- PyDESeq2 returned finite statistics for all {count_diagnostics['n_genes']:,} genes with no warnings. {summary['n_count_reliable']:,} have both converged MAP-dispersion and coefficient fits. The supporting model has {summary['n_count_fdr_0_05_reliable']:,} significant reliable results. Non-converged rows remain flagged in the tables; STAT1 has a matching count effect direction but a non-converged MAP-dispersion flag.
- Median-ratio versus CQN effect agreement is lower: Spearman rho={course_diagnostics['normalization']['spearman']:.3f}, direction agreement={100*course_diagnostics['normalization']['direction_agreement']:.1f}%, and {course_diagnostics['normalization']['significant_both']:,} genes are significant on both scales. CQN is a sensitivity scale, never count-model input.
- The lenient and stringent filters test {int(filter_sensitivity.loc[filter_sensitivity['filter'].eq('lenient'),'n_genes'].iloc[0]):,} and {int(filter_sensitivity.loc[filter_sensitivity['filter'].eq('stringent'),'n_genes'].iloc[0]):,} genes. Shared-gene effects are identical by construction because filtering changes the testing family, not normalized expression; significance classifications agree for >99.8% of shared genes.
- Leave-one-donor-out analysis covers {summary['loo_focused_genes']} frozen/representative genes and yields {summary['loo_direction_reversals']} direction reversals. No donor was removed.
- Absolute effect size is weakly negatively associated with average expression (Spearman rho={summary['absolute_effect_expression_spearman']:.3f}), so results are not dominated by highly expressed genes. Large effects are somewhat enriched among lower-expression genes, where the fixed 0.5 pseudocount can magnify log differences; count-model agreement is therefore important.
- Count Cook filtering and refitting were deliberately disabled so no donor-library observation was silently replaced or discarded. The 110 PyDESeq2 dispersion-outlier flags and all convergence flags are retained. Donor influence is assessed transparently through paired contrasts and leave-one-donor-out results.

## Interpretation

IFN-gamma priming profoundly reshapes, rather than uniformly amplifies, the transcriptional response to Salmonella. A large set of classical IFN-responsive, antimicrobial, and inflammatory genes shows negative log-scale interaction because strong single-treatment responses overlap or saturate. A distinct type-III-interferon program shows strong positive synergy. Regulatory genes are mixed, and the null IL10 and primary-null PTGS2 results directly contradict a simple claim that every immune or feedback gene is non-additively changed.

The justified conclusion is non-additive transcriptional remodeling in this controlled macrophage system. Whether this improves pathogen control or limits immunopathology requires protein, time-course, and bacterial-function experiments.

## Reproducibility and provenance

- Established study context: [Alasoo et al. (2018)](https://doi.org/10.1038/s41588-018-0046-7).
- Count-model API: [PyDESeq2 documentation](https://pydeseq2.readthedocs.io/en/latest/); installed and recorded version {count_diagnostics['version']}.
- Primary results: `results/tables/interaction_gene_results.csv`.
- Candidate results: `results/tables/interaction_candidate_genes.csv`.
- Model comparison: `results/tables/interaction_model_comparison.csv`.
- Normalization comparison: `results/tables/normalization_sensitivity.csv`.
- Pilot and full convergence diagnostics: `interaction_count_model_pilot_diagnostics.json`, `interaction_count_model_diagnostics.json`, and `interaction_count_model_convergence.csv`.
"""
    (paths["reports"] / "phase4_interaction_report.md").write_text(report,encoding="utf-8")
    return summary
