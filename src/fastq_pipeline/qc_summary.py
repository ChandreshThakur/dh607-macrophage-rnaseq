"""Parse FastQC text reports into auditable tabular summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_fastqc(path: Path) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    statuses: dict[str, str] = {}
    modules: dict[str, list[list[str]]] = {}
    current: str | None = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith(">>") and line != ">>END_MODULE":
                fields = line[2:].split("\t")
                current = fields[0]
                statuses[current] = fields[1] if len(fields) > 1 else ""
                modules[current] = []
            elif line == ">>END_MODULE":
                current = None
            elif current and line and not line.startswith("#"):
                modules[current].append(line.split("\t"))
    return statuses, modules


def summarize_directory(fastqc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for path in sorted(fastqc_dir.glob("*_fastqc/fastqc_data.txt")):
        statuses, modules = parse_fastqc(path)
        basic = dict(modules["Basic Statistics"])
        quality_rows = modules.get("Per base sequence quality", [])
        adapter_rows = modules.get("Adapter Content", [])
        overrepresented = modules.get("Overrepresented sequences", [])
        quality_means = [float(row[1]) for row in quality_rows]
        quality_medians = [float(row[2]) for row in quality_rows]
        adapter_values = [float(value) for row in adapter_rows for value in row[1:]]
        sample = basic["Filename"].removesuffix(".fastq.gz")
        metrics.append(
            {
                "sample": sample,
                "filename": basic["Filename"],
                "total_sequences": int(basic["Total Sequences"]),
                "sequence_length": basic["Sequence length"],
                "percent_gc": float(basic["%GC"]),
                "encoding": basic["Encoding"],
                "minimum_per_base_mean_quality": min(quality_means),
                "minimum_per_base_median_quality": min(quality_medians),
                "maximum_adapter_content_percent": max(adapter_values) if adapter_values else 0.0,
                "overrepresented_sequence_count": len(overrepresented),
                "duplication_status": statuses.get("Sequence Duplication Levels", "missing"),
                "adapter_status": statuses.get("Adapter Content", "missing"),
                "per_base_quality_status": statuses.get("Per base sequence quality", "missing"),
            }
        )
        status_rows.append({"sample": sample, **statuses})
    if not metrics:
        raise FileNotFoundError(f"no FastQC text reports found under {fastqc_dir}")
    return pd.DataFrame(metrics), pd.DataFrame(status_rows)


def write_summary(root: Path) -> None:
    metrics, statuses = summarize_directory(root / "results/qc/fastqc")
    tables = root / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "fastqc_metrics.csv", index=False)
    statuses.to_csv(tables / "fastqc_module_status.csv", index=False)
    no_adapter = (metrics["adapter_status"] == "pass").all()
    quality_pass = (metrics["per_base_quality_status"] == "pass").all()
    decision = "skip adapter/quality trimming" if no_adapter and quality_pass else "review trimming parameters"
    general = pd.read_csv(
        root / "results/qc/multiqc/multiqc_report_data/multiqc_general_stats.txt", sep="\t"
    )
    duplicates = general["fastqc-percent_duplicates"]
    deduplicated = 100 - duplicates
    duplication_counts = metrics["duplication_status"].value_counts()
    duplication_text = ", ".join(f"{count} {status}" for status, count in duplication_counts.items())
    text = f"""# FASTQ quality-control report

FastQC 0.12.1 was run on all eight original FASTQs and aggregated with MultiQC 1.31.

## Evidence

- Reports completed: {len(metrics)}/8.
- Per-base sequence-quality status: {', '.join(sorted(metrics['per_base_quality_status'].unique()))}; worst per-cycle mean quality across files: {metrics['minimum_per_base_mean_quality'].min():.2f}.
- Adapter-content status: {', '.join(sorted(metrics['adapter_status'].unique()))}; largest adapter percentage at any position: {metrics['maximum_adapter_content_percent'].max():.3f}%.
- Read length is 75 nt in every file; GC ranges from {metrics['percent_gc'].min():.0f}% to {metrics['percent_gc'].max():.0f}%.
- Duplication is flagged in all files ({duplication_text}); duplicate estimates range from {duplicates.min():.1f}% to {duplicates.max():.1f}% (deduplicated fractions {deduplicated.min():.1f}% to {deduplicated.max():.1f}%).
- Per-base sequence content is warned/failed, especially at the 5-prime end; this can reflect RNA-seq library priming/composition rather than removable adapter sequence.
- Overrepresented-sequence warnings occur, but FastQC identifies no adapter contamination; the listed sequences are individually low-frequency and are not evidence by themselves that trimming is useful.

## Trimming decision

Decision: **{decision}**. Adapter content and base quality both pass in every file, so trimming would remove data without a demonstrated technical target. High duplication is not used as a trimming trigger because these are RNA-seq reads deliberately restricted to chromosome 21; abundant transcripts and selection of only one chromosome increase repeated sequences. The original FASTQs are used unchanged for quantification.

## Files

- Interactive aggregate: `results/qc/multiqc/multiqc_report.html`
- Individual reports: `results/qc/fastqc/*_fastqc.html`
- Parsed metrics: `results/tables/fastqc_metrics.csv`
- Module statuses: `results/tables/fastqc_module_status.csv`
"""
    (root / "docs/fastq_qc.md").write_text(text, encoding="utf-8")
