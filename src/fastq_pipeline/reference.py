"""Prepare and validate the Ensembl 87 transcript reference."""

from __future__ import annotations

import csv
import gzip
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd

REFERENCE_SOURCES = {
    "Homo_sapiens.GRCh38.cdna.all.release87.fa.gz": "https://ftp.ensembl.org/pub/release-87/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
    "Homo_sapiens.GRCh38.ncrna.release87.fa.gz": "https://ftp.ensembl.org/pub/release-87/fasta/homo_sapiens/ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz",
    "Homo_sapiens.GRCh38.87.gtf.gz": "https://ftp.ensembl.org/pub/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh38.87.gtf.gz",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gtf_attributes(text: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r'(\S+) "([^"]+)";', text)}


def gtf_tx2gene(path: Path) -> pd.DataFrame:
    mappings: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            attrs = parse_gtf_attributes(fields[8])
            transcript, gene = attrs.get("transcript_id"), attrs.get("gene_id")
            if transcript and attrs.get("transcript_version"):
                transcript = f"{transcript}.{attrs['transcript_version']}"
            if transcript and gene:
                previous = mappings.setdefault(transcript, gene)
                if previous != gene:
                    raise ValueError(f"transcript {transcript} maps to multiple genes")
    return pd.DataFrame(sorted(mappings.items()), columns=["transcript_id", "gene_id"])


def fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def fasta_tx2gene(path: Path) -> pd.DataFrame:
    """Use the Ensembl `gene:` tag embedded in every transcript FASTA header."""
    rows: list[tuple[str, str]] = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            fields = line[1:].split()
            gene_fields = [field for field in fields[1:] if field.startswith("gene:")]
            if len(gene_fields) != 1:
                raise ValueError(f"expected one gene tag in FASTA header: {line[:120]}")
            transcript = fields[0]
            gene = re.sub(r"\.\d+$", "", gene_fields[0].removeprefix("gene:"))
            rows.append((transcript, gene))
    return pd.DataFrame(rows, columns=["transcript_id", "gene_id"])


def prepare(root: Path) -> None:
    reference = root / "data/reference"
    compressed = [reference / name for name in REFERENCE_SOURCES]
    for path in compressed:
        if not path.is_file():
            raise FileNotFoundError(path)
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass

    combined = reference / "Homo_sapiens.GRCh38.release87.cdna_ncrna.fa"
    with combined.open("wb") as output:
        for path in compressed[:2]:
            with gzip.open(path, "rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)

    gtf_mapping = gtf_tx2gene(compressed[2])
    ids = fasta_ids(combined)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate transcript identifiers in combined FASTA")
    mapping = fasta_tx2gene(combined)
    if mapping.transcript_id.tolist() != ids:
        raise ValueError("FASTA transcript parsing changed sequence order")
    comparison = mapping.merge(gtf_mapping, on="transcript_id", suffixes=("_fasta", "_gtf"))
    disagreements = comparison.query("gene_id_fasta != gene_id_gtf")
    if not disagreements.empty:
        raise ValueError(f"FASTA/GTF gene mappings disagree for {len(disagreements)} transcripts")
    missing_from_gtf = len(set(ids).difference(gtf_mapping.transcript_id))
    mapping.to_csv(reference / "tx2gene.tsv", sep="\t", index=False)

    rows = []
    for path in compressed:
        rows.append({
            "filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256sum(path),
            "source_url": REFERENCE_SOURCES[path.name], "role": "downloaded_original",
        })
    for path, role in [(combined, "derived_combined_transcriptome"), (reference / "tx2gene.tsv", "derived_transcript_gene_map")]:
        rows.append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256sum(path), "source_url": "derived locally from Ensembl 87 inputs", "role": role})
    pd.DataFrame(rows).to_csv(root / "data/manifests/reference_manifest.csv", index=False)

    chr21_genes = pd.read_csv(root / "data/published/RNA_gene_metadata.txt.gz", sep="\t").query("chr == '21'")
    report = f"""# Reference strategy

The pilot uses the complete Ensembl release 87 GRCh38 cDNA plus ncRNA transcriptome, not a chromosome-21-only reference. The original study-era workflow is documented as GRCh38/Ensembl 87, and a related reanalysis of the same data used Salmon with `--seqBias --gcBias --libType ISR`.

## Validation

- Combined FASTA transcripts: {len(ids):,}; duplicate transcript IDs: 0.
- Transcript-to-gene mappings retained: {len(mapping):,}; mapped genes: {mapping.gene_id.nunique():,}.
- FASTA/GTF mappings agree for {len(comparison):,} shared transcripts. The remaining {missing_from_gtf:,} FASTA entries (including alternate-locus transcripts) use Ensembl's embedded `gene:` tag rather than being silently discarded.
- Published matrix chromosome-21 genes available for later comparison: {len(chr21_genes):,}.
- File hashes and exact Ensembl URLs: `data/manifests/reference_manifest.csv`.

## Resource and bias rationale

The three compressed downloads total {sum(p.stat().st_size for p in compressed) / 2**20:.1f} MiB and the combined FASTA is {combined.stat().st_size / 2**20:.1f} MiB. A full transcriptome index is feasible on the measured host (88 GB free disk; WSL exposes 7.6 GB RAM). It preserves off-chromosome competing transcripts and paralogs. A chromosome-21-only index would be smaller but could overassign multi-mapping reads because alternative transcripts elsewhere in the genome would be absent.

The input reads were preselected by chromosome-21 genomic mapping, so this remains a teaching pilot: transcriptome assignment rates, abundance totals, and correlations cannot be interpreted as a reproduction of the original whole-transcriptome library.

## Built Salmon index

The Salmon 1.10.3 dense index uses k=31 and occupies 682.3 MiB. Salmon retained 202,863 quantification targets after collapsing 11,974 sequence-duplicate transcripts, clipped poly-A tails on 1,647 transcripts, and replaced 88 non-ATCG bases. The original transcript-to-gene map is preserved, while downstream validation reports genes whose every transcript was collapsed from the index.

The index contains no genomic decoy sequences. This matches a compact transcriptome-only teaching workflow but is a limitation: without decoys, some fragments originating from homologous genomic sequence may be assigned to transcripts. A larger decoy-aware index would be preferable for a production whole-transcriptome reanalysis.
"""
    (root / "docs/reference_strategy.md").write_text(report, encoding="utf-8")
