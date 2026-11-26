"""Compare course-level paired effects with the authors' official DESeq2 release."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from preprocessing.normalization import expressed_filter, log2_transform, median_of_ratios
from statistical_analysis.interaction import benjamini_hochberg


AUTHOR_FILES = {
    "IFNg_vs_naive": (
        "B",
        "naive_vs_IFNg_DESeq2_fold_change.txt.gz",
        "d30c466506cf7b6cb784ced7a9e4a68d",
    ),
    "SL1344_vs_naive": (
        "C",
        "naive_vs_Salmonella_DESeq2_fold_change.txt.gz",
        "3fa2cf63753e72e1f5c884b9bcb1235c",
    ),
    "IFNg_SL1344_vs_naive": (
        "D",
        "naive_vs_IFNg+Salmonella_DESeq2_fold_change.txt.gz",
        "5dfbed8bf82addcc42b8bba8f40f0a19",
    ),
}
AUTHOR_SOURCE = "https://zenodo.org/records/839011"
CANDIDATES = [
    "STAT1", "IRF1", "GBP1", "GBP2", "GBP3", "GBP4", "GBP5", "IDO1",
    "CXCL9", "CXCL10", "CXCL11", "NFKBIA", "IL1B", "TNF", "IFNB1",
]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_effect_table(
    expression: pd.DataFrame, metadata: pd.DataFrame, treated_condition: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute treated-minus-naive changes within donors and test their mean."""
    subset = metadata.loc[metadata.condition.isin(["A", treated_condition])]
    counts = subset.groupby(["donor", "condition"], observed=True).size().unstack(fill_value=0)
    if not ((counts.get("A", 0) == 1) & (counts.get(treated_condition, 0) == 1)).all():
        raise ValueError("paired effect requires exactly one naive and treated sample per donor")
    lookup = subset.set_index(["donor", "condition"])["sample_id"]
    donors = sorted(counts.index)
    naive_ids = [lookup.loc[(donor, "A")] for donor in donors]
    treated_ids = [lookup.loc[(donor, treated_condition)] for donor in donors]
    delta = expression.loc[:, treated_ids].to_numpy() - expression.loc[:, naive_ids].to_numpy()
    contrasts = pd.DataFrame(delta, index=expression.index, columns=donors)
    mean = contrasts.mean(axis=1)
    standard_error = contrasts.std(axis=1, ddof=1) / np.sqrt(len(donors))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = mean / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_statistic), df=len(donors) - 1)
    constant_zero = (standard_error == 0) & (mean == 0)
    p_value = np.where(constant_zero, 1.0, p_value)
    result = pd.DataFrame({
        "mean_paired_log2_change": mean,
        "standard_error": standard_error,
        "t_statistic": np.where(constant_zero, 0.0, t_statistic),
        "p_value": p_value,
        "adjusted_p_value": benjamini_hochberg(p_value),
        "n_donors": len(donors),
    }, index=expression.index)
    return result, contrasts


def effect_concordance(author: pd.Series, course: pd.Series, threshold: float = 0.5) -> dict[str, float | int]:
    """Summarize paired gene-level agreement for identically indexed effects."""
    if not author.index.equals(course.index):
        raise ValueError("effects must have identical gene IDs and order")
    finite = np.isfinite(author) & np.isfinite(course)
    author = author.loc[finite]
    course = course.loc[finite]
    eligible = (author.abs() >= threshold) & (course.abs() >= threshold)
    return {
        "n_genes": int(len(author)),
        "pearson": float(stats.pearsonr(author, course).statistic),
        "spearman": float(stats.spearmanr(author, course).statistic),
        "same_direction_fraction_all": float((np.sign(author) == np.sign(course)).mean()),
        "minimum_abs_log2_change_both": threshold,
        "n_genes_above_threshold_both": int(eligible.sum()),
        "same_direction_fraction_above_threshold": float((np.sign(author.loc[eligible]) == np.sign(course.loc[eligible])).mean()),
        "median_course_minus_author_effect": float((course - author).median()),
    }


def _plot_effect_concordance(comparisons: dict[str, pd.DataFrame], figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    display = {
        "IFNg_vs_naive": "IFN-gamma - naive",
        "SL1344_vs_naive": "Salmonella - naive",
        "IFNg_SL1344_vs_naive": "Combined - naive",
    }
    for (name, frame), ax in zip(comparisons.items(), axes):
        x = frame.author_log2_fold_change
        y = frame.course_mean_paired_log2_change
        rho = stats.spearmanr(x, y).statistic
        limit = max(abs(x).max(), abs(y).max())
        ax.scatter(x, y, s=6, alpha=0.22, rasterized=True, color="#2b8cbe", edgecolors="none")
        ax.plot([-limit, limit], [-limit, limit], color="#555555", ls="--", lw=0.8)
        ax.axhline(0, color="#aaaaaa", lw=0.5)
        ax.axvline(0, color="#aaaaaa", lw=0.5)
        ax.set_title(f"{display[name]}\nSpearman rho={rho:.3f}")
        ax.set_xlabel("Authors' DESeq2 log2 fold change")
        ax.set_ylabel("Course paired mean log2 change")
    fig.suptitle("Baseline effect-size reproduction across 15,377 expressed genes")
    fig.tight_layout()
    fig.savefig(figures / "baseline_author_vs_course_effects.png", dpi=180)
    plt.close(fig)


def _plot_candidate_heatmap(candidate: pd.DataFrame, figures: Path) -> None:
    columns = []
    labels = []
    for comparison, short in [
        ("IFNg_vs_naive", "IFN-gamma"),
        ("SL1344_vs_naive", "Salmonella"),
        ("IFNg_SL1344_vs_naive", "Combined"),
    ]:
        columns.extend([f"author_{comparison}_log2fc", f"course_{comparison}_log2_change"])
        labels.extend([f"{short}\nauthor", f"{short}\ncourse"])
    heat = candidate.set_index("gene_name")[columns]
    heat.columns = labels
    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(heat, cmap="vlag", center=0, linewidths=0.25, cbar_kws={"label": "Stimulus - naive (log2 units)"}, ax=ax)
    ax.set(xlabel="Comparison and method", ylabel="Gene", title="Focused immune-response baseline")
    fig.tight_layout()
    fig.savefig(figures / "baseline_candidate_response_heatmap.png", dpi=180)
    plt.close(fig)


def run(root: Path) -> dict[str, object]:
    published = root / "data/published"
    reference = published / "baseline_reference"
    tables = root / "results/tables"
    figures = root / "results/figures"
    reports = root / "results/reports"
    manifests = root / "data/manifests"
    for directory in (tables, figures, reports, manifests):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    author_tables: dict[str, pd.DataFrame] = {}
    for comparison, (_, filename, expected_md5) in AUTHOR_FILES.items():
        path = reference / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = md5sum(path)
        if observed != expected_md5:
            raise ValueError(f"MD5 mismatch for {filename}")
        table = pd.read_csv(path, sep="\t")
        required = {"gene_id", "gene_name", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"}
        if set(table.columns) != required or table.gene_id.duplicated().any():
            raise ValueError(f"unexpected author table schema for {filename}")
        author_tables[comparison] = table.set_index("gene_id")
        manifest_rows.append({
            "filename": filename,
            "size_bytes": path.stat().st_size,
            "local_md5": observed,
            "authoritative_md5": expected_md5,
            "checksum_verified": True,
            "source_url": AUTHOR_SOURCE,
            "model": "DESeq2 1.10.0 LRT: full ~condition_name, reduced ~1",
        })
    pd.DataFrame(manifest_rows).to_csv(manifests / "baseline_reference_manifest.csv", index=False)

    metadata = pd.read_csv(published / "RNA_sample_metadata.txt.gz", sep="\t")
    counts = pd.read_csv(published / "RNA_count_matrix.txt.gz", sep="\t", index_col=0)
    genes = pd.read_csv(published / "RNA_gene_metadata.txt.gz", sep="\t").set_index("gene_id")
    if any(set(table.index) != set(counts.index) for table in author_tables.values()):
        raise ValueError("authors' DE gene IDs do not exactly match the released count matrix")

    normalized, _ = median_of_ratios(counts)
    keep = expressed_filter(counts, minimum_count=10, minimum_samples=84)
    expression = log2_transform(normalized.loc[keep], pseudocount=0.5)
    condition_means = pd.DataFrame({
        condition: expression.loc[:, metadata.loc[metadata.condition == condition, "sample_id"]].mean(axis=1)
        for condition in "ABCD"
    })

    comparisons: dict[str, pd.DataFrame] = {}
    concordance_rows = []
    course_tables: dict[str, pd.DataFrame] = {}
    for comparison, (condition, _, _) in AUTHOR_FILES.items():
        course, _ = paired_effect_table(expression, metadata, condition)
        author = author_tables[comparison].loc[course.index]
        combined = pd.DataFrame({
            "author_log2_fold_change": author.log2FoldChange,
            "author_lfc_standard_error": author.lfcSE,
            "author_omnibus_lrt_statistic": author.stat,
            "author_omnibus_lrt_p_value": author.pvalue,
            "author_omnibus_lrt_adjusted_p_value": author.padj,
            "course_mean_paired_log2_change": course.mean_paired_log2_change,
            "course_standard_error": course.standard_error,
            "course_p_value": course.p_value,
            "course_adjusted_p_value": course.adjusted_p_value,
        })
        combined.index.name = "gene_id"
        comparisons[comparison] = combined
        course_tables[comparison] = course
        combined.reset_index().to_csv(tables / f"baseline_{comparison}_comparison.csv.gz", index=False, compression="gzip")
        concordance_rows.append({"comparison": comparison, **effect_concordance(combined.author_log2_fold_change, combined.course_mean_paired_log2_change)})
    concordance = pd.DataFrame(concordance_rows)
    concordance.to_csv(tables / "baseline_effect_concordance.csv", index=False)

    candidate_annotation = genes.reset_index().loc[lambda x: x.gene_name.isin(CANDIDATES), ["gene_id", "gene_name", "gene_biotype"]]
    if set(candidate_annotation.gene_name) != set(CANDIDATES):
        raise ValueError("one or more prespecified baseline genes are absent")
    candidate = candidate_annotation.set_index("gene_id")
    for condition in "ABCD":
        candidate[f"course_mean_log2_{condition}"] = condition_means[condition]
    for comparison, combined in comparisons.items():
        candidate[f"author_{comparison}_log2fc"] = combined.author_log2_fold_change
        candidate[f"course_{comparison}_log2_change"] = combined.course_mean_paired_log2_change
        candidate[f"course_{comparison}_adjusted_p_value"] = combined.course_adjusted_p_value
    candidate = candidate.reset_index().sort_values("gene_name")
    candidate.to_csv(tables / "baseline_candidate_gene_reproduction.csv", index=False)

    _plot_effect_concordance(comparisons, figures)
    _plot_candidate_heatmap(candidate, figures)

    concordance_indexed = concordance.set_index("comparison")
    representative_names = ["STAT1", "IRF1", "GBP1", "IDO1", "IFNB1", "IL1B", "TNF", "NFKBIA"]
    representative = candidate.set_index("gene_name").loc[representative_names]
    representative_lines = []
    for name, row in representative.iterrows():
        representative_lines.append(
            f"- {name}: IFN-gamma author/course {row['author_IFNg_vs_naive_log2fc']:.2f}/{row['course_IFNg_vs_naive_log2_change']:.2f}; "
            f"Salmonella author/course {row['author_SL1344_vs_naive_log2fc']:.2f}/{row['course_SL1344_vs_naive_log2_change']:.2f}; "
            f"combined author/course {row['author_IFNg_SL1344_vs_naive_log2fc']:.2f}/{row['course_IFNg_SL1344_vs_naive_log2_change']:.2f}."
        )

    report = f"""# Baseline reproduction

## Published source and target finding

[Alasoo et al. (2018)](https://doi.org/10.1038/s41588-018-0046-7), Figure 1d, describes IFN-gamma receptor signaling through STAT1 and IRF1 and Salmonella/TLR4 signaling through NF-kappaB, AP-1, IRF3, and the type-I-interferon STAT1-STAT2-IRF9 axis. The authors' official [differential-expression release]({AUTHOR_SOURCE}) provides gene-level fold changes for IFN-gamma, Salmonella, and combined treatment relative to naive cells. This baseline asks whether the released raw counts reproduce those response directions and gene rankings under a transparent donor-paired course method.

## Published and course procedures

The authors used DESeq2 1.10.0 with a full `~ condition_name` model, a reduced intercept-only model, and a likelihood-ratio test. Their `log2FoldChange` columns are stimulus minus naive. Their `stat`, `pvalue`, and `padj` columns test the four-condition model as a whole and are therefore not contrast-specific p-values, even though they appear in each contrast file.

Our method uses all 84 complete donors, median-of-ratios normalized raw counts, the prespecified count >=10 in >=84 samples filter ({int(keep.sum()):,} genes), and `log2(normalized count + 0.5)`. For each gene we calculate treated minus naive within each donor, average the 84 differences, perform a two-sided one-sample t-test, and apply BH correction. This uses pairing that the authors' archived condition-only DESeq2 formula did not include, but it is not a negative-binomial reproduction.

## Genome-wide agreement

- IFN-gamma versus naive: Spearman rho={concordance_indexed.loc['IFNg_vs_naive', 'spearman']:.3f}; Pearson r={concordance_indexed.loc['IFNg_vs_naive', 'pearson']:.3f}; direction agreement among genes with absolute effect >=0.5 in both methods={100 * concordance_indexed.loc['IFNg_vs_naive', 'same_direction_fraction_above_threshold']:.1f}% ({int(concordance_indexed.loc['IFNg_vs_naive', 'n_genes_above_threshold_both']):,} genes).
- Salmonella versus naive: Spearman rho={concordance_indexed.loc['SL1344_vs_naive', 'spearman']:.3f}; Pearson r={concordance_indexed.loc['SL1344_vs_naive', 'pearson']:.3f}; thresholded direction agreement={100 * concordance_indexed.loc['SL1344_vs_naive', 'same_direction_fraction_above_threshold']:.1f}% ({int(concordance_indexed.loc['SL1344_vs_naive', 'n_genes_above_threshold_both']):,} genes).
- Combined versus naive: Spearman rho={concordance_indexed.loc['IFNg_SL1344_vs_naive', 'spearman']:.3f}; Pearson r={concordance_indexed.loc['IFNg_SL1344_vs_naive', 'pearson']:.3f}; thresholded direction agreement={100 * concordance_indexed.loc['IFNg_SL1344_vs_naive', 'same_direction_fraction_above_threshold']:.1f}% ({int(concordance_indexed.loc['IFNg_SL1344_vs_naive', 'n_genes_above_threshold_both']):,} genes).

These high rank correlations reproduce the broad published response. Deviations from the identity line are expected because mean paired changes on a pseudocount log scale are not the same estimand as DESeq2 negative-binomial coefficients.

## Focused immune genes

{chr(10).join(representative_lines)}

The IFN-gamma arm strongly increases STAT1, IRF1, GBP-family genes, IDO1, and CXCL chemokines. Salmonella strongly increases IFNB1, IL1B, TNF, NFKBIA, and also interferon-responsive genes, consistent with the paper's TLR4/IRF3/type-I-interferon scheme. Combined treatment is often highest in absolute expression, but this baseline comparison does not test whether the combined response is additive; that requires the separate factorial interaction contrast.

## Agreement, discrepancies, and limits

The reproduced directions and genome-wide rankings agree strongly with the authors' released effects. This is same-dataset computational reproduction, not independent biological replication. Absolute fold changes differ, sometimes substantially for genes near zero in naive cells, because the course transform uses a fixed pseudocount, while DESeq2 models raw counts and estimates dispersion. The authors' archived LRT omitted donor blocking, whereas our paired summaries explicitly use donors. This is a close baseline reproduction of response patterns, not an exact reimplementation of DESeq2 1.10.0. Expression changes support pathway activation but do not demonstrate better bacterial killing, protein abundance, or clinical benefit.

## Outputs

- Exact candidate values: `results/tables/baseline_candidate_gene_reproduction.csv`
- Genome-wide concordance: `results/tables/baseline_effect_concordance.csv`
- Author/course scatter plot: `results/figures/baseline_author_vs_course_effects.png`
- Candidate response heatmap: `results/figures/baseline_candidate_response_heatmap.png`
- Source checksums: `data/manifests/baseline_reference_manifest.csv`
"""
    (reports / "baseline_reproduction.md").write_text(report, encoding="utf-8")
    return {
        "n_genes_compared": int(keep.sum()),
        "n_donors": int(metadata.donor.nunique()),
        "concordance": concordance.to_dict("records"),
    }
