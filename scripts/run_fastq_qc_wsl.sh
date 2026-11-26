#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tools_root="${DH607_TOOLS_ROOT:-/mnt/c/tmp/dh607-tools}"
micromamba="${tools_root}/bin/micromamba"
environment="${DH607_ENV_PREFIX:-${HOME}/.local/share/dh607-env}"
fastqc_out="${project_root}/results/qc/fastqc"
multiqc_out="${project_root}/results/qc/multiqc"

mkdir -p "${fastqc_out}" "${multiqc_out}"
"${micromamba}" run --prefix "${environment}" fastqc \
  --threads 4 --outdir "${fastqc_out}" "${project_root}"/data/raw_fastq/*.fastq.gz
"${micromamba}" run --prefix "${environment}" multiqc \
  "${fastqc_out}" --outdir "${multiqc_out}" --force --filename multiqc_report.html
