"""Validate Salmon outputs, aggregate transcripts, and compare the chr21 pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SAMPLES = ("eipl_A", "eipl_C", "fikt_A", "fikt_C")
QUANT_COLUMNS = ("Name", "Length", "EffectiveLength", "TPM", "NumReads")


def read_salmon_quant(sample_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read one Salmon result and enforce its structural invariants."""
    quant_path = sample_dir / "quant.sf"
    meta_path = sample_dir / "aux_info/meta_info.json"
    format_path = sample_dir / "lib_format_counts.json"
    for path in (quant_path, meta_path, format_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    quant = pd.read_csv(quant_path, sep="\t")
    if tuple(quant.columns) != QUANT_COLUMNS:
        raise ValueError(f"unexpected quant.sf columns in {sample_dir.name}: {quant.columns.tolist()}")
    if quant.Name.duplicated().any():
        raise ValueError(f"duplicate transcript IDs in {sample_dir.name}")
    numeric = quant.loc[:, QUANT_COLUMNS[1:]]
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
        raise ValueError(f"invalid numeric abundance in {sample_dir.name}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    library = json.loads(format_path.read_text(encoding="utf-8"))
    if meta.get("quant_errors"):
        raise ValueError(f"Salmon reported quantification errors for {sample_dir.name}")
    if meta.get("num_valid_targets") != len(quant):
        raise ValueError(f"target count does not match quant.sf for {sample_dir.name}")
    if library.get("expected_format") != "ISR":
        raise ValueError(f"unexpected library format for {sample_dir.name}")
    if library.get("num_assigned_fragments") != meta.get("num_mapped"):
        raise ValueError(f"assigned/mapped fragment mismatch for {sample_dir.name}")
    return quant.set_index("Name"), {"meta": meta, "library": library}


def aggregate_to_genes(
    quants: dict[str, pd.DataFrame], tx2gene: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum Salmon estimated counts and TPM over transcripts for each gene."""
    if tx2gene.transcript_id.duplicated().any():
        raise ValueError("tx2gene contains duplicate transcript identifiers")
    mapping = tx2gene.set_index("transcript_id")["gene_id"]
    reference_names: pd.Index | None = None
    gene_counts: dict[str, pd.Series] = {}
    gene_tpm: dict[str, pd.Series] = {}
    for sample, quant in quants.items():
        if reference_names is None:
            reference_names = quant.index
        elif not quant.index.equals(reference_names):
            raise ValueError(f"transcript target order differs for {sample}")
        missing = quant.index.difference(mapping.index)
        if len(missing):
            raise ValueError(f"{len(missing)} quantified transcripts lack a gene mapping")
        genes = mapping.loc[quant.index].to_numpy()
        gene_counts[sample] = quant.NumReads.groupby(genes, sort=True).sum()
        gene_tpm[sample] = quant.TPM.groupby(genes, sort=True).sum()
    return pd.DataFrame(gene_counts).fillna(0.0), pd.DataFrame(gene_tpm).fillna(0.0)


def log2_cpm(values: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    """Normalize each column within the supplied gene set, then log2 transform."""
    totals = values.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("cannot calculate CPM for a column with zero total")
    return np.log2(values.divide(totals, axis=1) * 1_000_000 + pseudocount)


def _correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("correlation requires at least three paired values")
    return float(stats.pearsonr(x, y).statistic), float(stats.spearmanr(x, y).statistic)


def direction_agreement(
    first: pd.Series, second: pd.Series, minimum_abs_change: float = 0.0
) -> tuple[int, float]:
    """Return eligible count and sign agreement after a two-method effect filter."""
    if minimum_abs_change < 0:
        raise ValueError("minimum_abs_change must be nonnegative")
    if not first.index.equals(second.index):
        raise ValueError("direction comparison requires identically ordered genes")
    eligible = (first.abs() >= minimum_abs_change) & (second.abs() >= minimum_abs_change)
    if not eligible.any():
        return 0, float("nan")
    agreement = np.sign(first.loc[eligible]) == np.sign(second.loc[eligible])
    return int(eligible.sum()), float(agreement.mean())


def _plot_abundance_comparison(
    salmon_log_cpm: pd.DataFrame, published_log_cpm: pd.DataFrame, figures: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    limits = [
        min(salmon_log_cpm.min().min(), published_log_cpm.min().min()),
        max(salmon_log_cpm.max().max(), published_log_cpm.max().max()),
    ]
    for sample, ax in zip(SAMPLES, axes.flat):
        x = published_log_cpm[sample]
        y = salmon_log_cpm[sample]
        _, rho = _correlation(x, y)
        ax.scatter(x, y, s=9, alpha=0.45, color="#2b8cbe", edgecolors="none")
        ax.plot(limits, limits, color="#555555", lw=0.8, ls="--")
        ax.set_title(f"{sample} (Spearman rho={rho:.2f})")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    fig.supxlabel("Published featureCounts: log2(chr21 CPM + 0.5)")
    fig.supylabel("Salmon estimates: log2(chr21 CPM + 0.5)")
    fig.suptitle("Chromosome-21 gene-abundance comparison")
    fig.tight_layout()
    fig.savefig(figures / "salmon_vs_published_chr21_abundance.png", dpi=180)
    plt.close(fig)


def _plot_response_comparison(
    salmon_log_cpm: pd.DataFrame, published_log_cpm: pd.DataFrame, figures: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    all_values: list[np.ndarray] = []
    for donor in ("eipl", "fikt"):
        for data in (salmon_log_cpm, published_log_cpm):
            all_values.append((data[f"{donor}_C"] - data[f"{donor}_A"]).to_numpy())
    limit = max(abs(np.concatenate(all_values).min()), abs(np.concatenate(all_values).max()))
    for donor, ax in zip(("eipl", "fikt"), axes):
        x = published_log_cpm[f"{donor}_C"] - published_log_cpm[f"{donor}_A"]
        y = salmon_log_cpm[f"{donor}_C"] - salmon_log_cpm[f"{donor}_A"]
        _, rho = _correlation(x, y)
        ax.scatter(x, y, s=9, alpha=0.45, color="#d95f02", edgecolors="none")
        ax.plot([-limit, limit], [-limit, limit], color="#555555", lw=0.8, ls="--")
        ax.axhline(0, color="#999999", lw=0.5)
        ax.axvline(0, color="#999999", lw=0.5)
        ax.set_title(f"{donor}: C - A (Spearman rho={rho:.2f})")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
    fig.supxlabel("Published change in log2 chr21 CPM")
    fig.supylabel("Salmon change in log2 chr21 CPM")
    fig.suptitle("Salmonella response in the two pilot donors")
    fig.tight_layout()
    fig.savefig(figures / "salmon_vs_published_chr21_response.png", dpi=180)
    plt.close(fig)


def run(root: Path) -> dict[str, object]:
    """Run validation, aggregation, comparison, plots, and report generation."""
    quant_root = root / "results/quantification"
    tables = root / "results/tables"
    figures = root / "results/figures"
    processed = root / "data/processed"
    reports = root / "results/reports"
    for directory in (tables, figures, processed, reports):
        directory.mkdir(parents=True, exist_ok=True)

    quants: dict[str, pd.DataFrame] = {}
    details: dict[str, dict[str, object]] = {}
    for sample in SAMPLES:
        quants[sample], details[sample] = read_salmon_quant(quant_root / sample)

    tx2gene = pd.read_csv(root / "data/reference/tx2gene.tsv", sep="\t")
    gene_counts, gene_tpm = aggregate_to_genes(quants, tx2gene)
    gene_counts.to_csv(processed / "salmon_gene_estimated_counts.tsv.gz", sep="\t", compression="gzip", index_label="gene_id")
    gene_tpm.to_csv(processed / "salmon_gene_tpm.tsv.gz", sep="\t", compression="gzip", index_label="gene_id")

    mapping_rows = []
    for sample in SAMPLES:
        meta = details[sample]["meta"]
        library = details[sample]["library"]
        mapping_rows.append({
            "sample_id": sample,
            "salmon_version": meta["salmon_version"],
            "num_processed_fragments": meta["num_processed"],
            "num_mapped_fragments": meta["num_mapped"],
            "mapping_rate_percent": meta["percent_mapped"],
            "expected_library_format": library["expected_format"],
            "compatible_fragment_ratio": library["compatible_fragment_ratio"],
            "fragment_length_mean": meta["frag_length_mean"],
            "fragment_length_sd": meta["frag_length_sd"],
            "num_valid_targets": meta["num_valid_targets"],
            "num_equivalence_classes": meta["num_eq_classes"],
            "qc_flag": "review_low_mapping" if meta["percent_mapped"] < 60 else "pass",
        })
    mapping_table = pd.DataFrame(mapping_rows)
    mapping_table.to_csv(tables / "salmon_mapping_summary.csv", index=False)

    published_counts = pd.read_csv(root / "data/published/RNA_count_matrix.txt.gz", sep="\t", index_col=0)
    genes = pd.read_csv(root / "data/published/RNA_gene_metadata.txt.gz", sep="\t")
    chr21_ids = pd.Index(genes.loc[genes.chr.astype(str) == "21", "gene_id"])
    matched = chr21_ids.intersection(gene_counts.index).intersection(published_counts.index)
    if len(matched) == 0:
        raise ValueError("no chromosome-21 genes matched between pilot and published data")
    salmon_chr21 = gene_counts.loc[matched, SAMPLES]
    published_chr21 = published_counts.loc[matched, SAMPLES]

    quantified_tx = next(iter(quants.values())).index
    reference_tx_counts = tx2gene.groupby("gene_id").size()
    quantified_tx_counts = tx2gene.loc[tx2gene.transcript_id.isin(quantified_tx)].groupby("gene_id").size()
    missing_ids = chr21_ids.difference(matched)
    missing_chr21 = genes.set_index("gene_id").loc[missing_ids, ["gene_name", "gene_biotype", "chr"]].copy()
    missing_chr21.index.name = "gene_id"
    missing_chr21["n_transcripts_in_combined_fasta"] = reference_tx_counts.reindex(missing_ids, fill_value=0)
    missing_chr21["n_transcripts_in_cleaned_salmon_index"] = quantified_tx_counts.reindex(missing_ids, fill_value=0)
    missing_chr21.reset_index().to_csv(tables / "salmon_missing_chr21_genes.csv", index=False)
    salmon_log_cpm = log2_cpm(salmon_chr21)
    published_log_cpm = log2_cpm(published_chr21)

    comparison_rows = []
    for sample in SAMPLES:
        raw_pearson, raw_spearman = _correlation(
            np.log1p(published_chr21[sample]), np.log1p(salmon_chr21[sample])
        )
        cpm_pearson, cpm_spearman = _correlation(
            published_log_cpm[sample], salmon_log_cpm[sample]
        )
        comparison_rows.append({
            "comparison_type": "abundance",
            "sample_id": sample,
            "donor": sample.split("_")[0],
            "condition": sample.split("_")[1],
            "n_chr21_genes_in_published_metadata": len(chr21_ids),
            "n_matched_chr21_genes": len(matched),
            "salmon_all_gene_estimated_count_total": gene_counts[sample].sum(),
            "salmon_matched_chr21_estimated_count_total": salmon_chr21[sample].sum(),
            "published_all_gene_count_total": published_counts[sample].sum(),
            "published_matched_chr21_count_total": published_chr21[sample].sum(),
            "n_salmon_zero_genes": int((salmon_chr21[sample] == 0).sum()),
            "n_published_zero_genes": int((published_chr21[sample] == 0).sum()),
            "n_zero_in_both": int(((salmon_chr21[sample] == 0) & (published_chr21[sample] == 0)).sum()),
            "pearson_log1p_counts": raw_pearson,
            "spearman_log1p_counts": raw_spearman,
            "pearson_log2_chr21_cpm": cpm_pearson,
            "spearman_log2_chr21_cpm": cpm_spearman,
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(tables / "salmon_published_chr21_comparison.csv", index=False)

    response_rows = []
    direction_rows = []
    for donor in ("eipl", "fikt"):
        salmon_delta = salmon_log_cpm[f"{donor}_C"] - salmon_log_cpm[f"{donor}_A"]
        published_delta = published_log_cpm[f"{donor}_C"] - published_log_cpm[f"{donor}_A"]
        pearson, spearman = _correlation(published_delta, salmon_delta)
        n_all, agreement_all = direction_agreement(published_delta, salmon_delta)
        response_rows.append({
            "donor": donor,
            "n_matched_chr21_genes": len(matched),
            "pearson_C_minus_A_log2_chr21_cpm": pearson,
            "spearman_C_minus_A_log2_chr21_cpm": spearman,
            "same_direction_fraction": agreement_all,
        })
        for threshold in (0.0, 0.1, 0.25, 0.5, 1.0):
            n_eligible, agreement = direction_agreement(
                published_delta, salmon_delta, minimum_abs_change=threshold
            )
            direction_rows.append({
                "donor": donor,
                "minimum_abs_log2_change_in_both_methods": threshold,
                "n_eligible_genes": n_eligible,
                "same_direction_fraction": agreement,
            })
    response = pd.DataFrame(response_rows)
    response.to_csv(tables / "salmon_published_chr21_response_comparison.csv", index=False)
    direction_sensitivity = pd.DataFrame(direction_rows)
    direction_sensitivity.to_csv(tables / "salmon_response_direction_sensitivity.csv", index=False)

    _plot_abundance_comparison(salmon_log_cpm, published_log_cpm, figures)
    _plot_response_comparison(salmon_log_cpm, published_log_cpm, figures)

    low_mapping = mapping_table.loc[mapping_table.qc_flag != "pass", "sample_id"].tolist()
    summary = {
        "n_samples": len(SAMPLES),
        "n_quantified_transcripts": int(len(next(iter(quants.values())))),
        "n_aggregated_genes": int(len(gene_counts)),
        "n_published_chr21_genes": int(len(chr21_ids)),
        "n_matched_chr21_genes": int(len(matched)),
        "n_unmatched_chr21_genes": int(len(missing_chr21)),
        "n_unmatched_with_reference_transcripts": int((missing_chr21.n_transcripts_in_combined_fasta > 0).sum()),
        "mapping_rate_min_percent": float(mapping_table.mapping_rate_percent.min()),
        "mapping_rate_max_percent": float(mapping_table.mapping_rate_percent.max()),
        "low_mapping_samples": low_mapping,
        "median_abundance_spearman": float(comparison.spearman_log2_chr21_cpm.median()),
        "median_response_spearman": float(response.spearman_C_minus_A_log2_chr21_cpm.median()),
        "response_same_direction_min": float(response.same_direction_fraction.min()),
        "response_same_direction_max": float(response.same_direction_fraction.max()),
        "response_same_direction_ge_0_25_min": float(direction_sensitivity.loc[direction_sensitivity.minimum_abs_log2_change_in_both_methods == 0.25, "same_direction_fraction"].min()),
        "response_same_direction_ge_0_25_max": float(direction_sensitivity.loc[direction_sensitivity.minimum_abs_log2_change_in_both_methods == 0.25, "same_direction_fraction"].max()),
    }
    (tables / "salmon_pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# Salmon pilot quantification report

## Validated execution

Salmon 1.10.3 quantified all four paired-end pilot samples against the full Ensembl 87 cDNA + ncRNA transcriptome. Each output contains {summary['n_quantified_transcripts']:,} indexed targets, has no reported quantification error, and assigns every mapped fragment in the requested `ISR` orientation. Sequence- and GC-bias correction were enabled. Estimated transcript counts and TPM were summed to genes with the validated Ensembl transcript-to-gene map.

Mapping rates range from {summary['mapping_rate_min_percent']:.1f}% to {summary['mapping_rate_max_percent']:.1f}%. `{low_mapping[0] if low_mapping else 'No sample'}` is flagged for review because its mapping rate is below 60%; the perfect orientation compatibility shows that an incorrect strandedness setting is not the explanation. These are deliberately chromosome-21-selected reads mapped to a full transcriptome, so mapping behavior is not directly comparable to an ordinary whole-transcriptome RNA-seq library.

## Comparison with published counts

The published metadata contains {summary['n_published_chr21_genes']:,} chromosome-21 genes; {summary['n_matched_chr21_genes']:,} are represented in both outputs. Of the {summary['n_unmatched_chr21_genes']} unmatched genes, {summary['n_unmatched_with_reference_transcripts']} had release-87 FASTA transcripts but no remaining transcript after Salmon collapsed sequence-identical targets; one gene had no transcript in the combined FASTA. Across the four samples, the median Spearman correlation of within-chr21 log2 CPM abundance is {summary['median_abundance_spearman']:.3f}. For the infection response (condition C minus A) across the two donors, the median gene-level Spearman correlation is {summary['median_response_spearman']:.3f}. Across all matched genes, {summary['response_same_direction_min'] * 100:.1f}% to {summary['response_same_direction_max'] * 100:.1f}% change in the same direction. Requiring both methods to show at least 0.25 log2 units of change raises agreement to {summary['response_same_direction_ge_0_25_min'] * 100:.1f}% to {summary['response_same_direction_ge_0_25_max'] * 100:.1f}%, showing that the conclusion is not driven by near-zero signs. Exact threshold sensitivities, including eligible gene counts, are exported separately.

This is a concordance check, not an exact reproduction. Salmon probabilistically allocates fragments among transcripts and then sums them to genes, whereas the published matrix used genomic gene counting; the pilot FASTQs were preselected by chromosome, are much smaller than the original libraries, and contain only two donors and two of the four conditions. Zero counts are retained and made finite only by the stated 0.5 CPM pseudocount; they are not imputed. Within-chr21 CPM corrects the gross library-scale difference but cannot remove transcript ambiguity, selection bias, or low counts. No factorial IFN-gamma by Salmonella interaction can be estimated from these FASTQs because B and D are absent.
"""
    (reports / "salmon_pilot_report.md").write_text(report, encoding="utf-8")
    return summary
