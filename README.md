# DH607 macrophage transcriptomics

This project reanalyzes the public RNA-seq data from Alasoo et al. (2018) to test how prior IFN-γ exposure changes the human macrophage transcriptional response to *Salmonella* challenge. The cohort has 84 donors with four conditions each. The primary analysis tests `(D − B) − (C − A)` within each donor on the normalized log₂ scale. A donor-blocked negative-binomial model and an alternative normalization provide sensitivity checks. See the [statistical design](docs/phase4_statistical_design.md) and the [original study](https://doi.org/10.1038/s41588-018-0046-7).

[Final presentation](presentation/DH607_final_presentation_updated.pptx)

## Setup

Use Python 3.11 or newer for the matrix analysis and tests. From the repository root, install the project and its test dependencies.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

`environment.yml` also defines a Conda environment with the external tools for the optional raw-read pilot. The tests that check full-cohort outputs run once their input data and generated files are present.

## Public inputs

Keep the original compressed filenames. The manifests under `data/manifests/` record source URLs and checksums.

| Input | Local directory | Release |
| --- | --- | --- |
| `RNA_count_matrix.txt.gz`, `RNA_cqn_matrix.txt.gz`, `RNA_gene_metadata.txt.gz`, `RNA_sample_metadata.txt.gz` | `data/published/` | [RNA-seq matrices and metadata](https://zenodo.org/records/1188300) |
| The three `naive_vs_*_DESeq2_fold_change.txt.gz` tables | `data/published/baseline_reference/` | [Published baseline fold changes](https://zenodo.org/records/839011) |
| `go-basic.obo`, `goa_human.gaf.gz`, `Homo_sapiens.GRCh38.cdna.all.release79.fa.gz` | `data/reference/` | [Phase 5 reference manifest](data/manifests/phase5_reference_manifest.csv) |

For an exact enrichment reproduction, use reference files that match the SHA-256 values in the Phase 5 manifest. The ontology snapshot identifies itself as the 2026-07-26 release, and the human annotation header is dated 2026-05-21. The Gene Ontology `current` URLs can serve updated snapshots. The bundled ENA TSV is the public run metadata used by the audit.

## Run the analysis

```sh
python scripts/run_published_analysis.py
python scripts/run_phase3.py
python scripts/run_phase4.py
python scripts/run_phase5.py --permutations 500
python scripts/run_phase6.py
python -m pytest -q
```

The scripts generate exploratory summaries and baseline comparisons, the paired and count-model estimates, the sensitivity and ontology results, and the final audit. Tables and figures are written under `results/`. The 6,477 material interactions reported in the presentation pass both adjusted-p-value screens, agree in sign, have a reliable count fit, and meet the 0.5 log₂-unit effect cutoff. The study measures relative RNA abundance in the published cohort.

## Optional chromosome 21 pilot

Place the eight FASTQs in `data/raw_fastq/` and the downloaded Ensembl 87 references in `data/reference/`. This pilot uses two donors in conditions A and C. [Published pilot FASTQs](https://zenodo.org/records/1173306) · [Ensembl file names and checksums](data/manifests/reference_manifest.csv).

Run the following with FastQC, MultiQC, and Salmon 1.10.3 available in a Linux or WSL shell.

```sh
python scripts/run_data_audit.py
python scripts/prepare_reference.py
python scripts/run_fastq_qc.py
python scripts/summarize_fastqc.py
salmon index -t data/reference/Homo_sapiens.GRCh38.release87.cdna_ncrna.fa -i data/reference/salmon_index_ensembl87 -k 31
bash scripts/run_salmon_pilot_wsl.sh
python scripts/summarize_salmon_pilot.py
```

The ENA audit in the matrix workflow uses saved run metadata. The pilot workflow uses the separate training release listed above.
