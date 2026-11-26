#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
salmon="${project_root}/scripts/salmon_wsl.sh"
index="${project_root}/data/reference/salmon_index_ensembl87"
fastq="${project_root}/data/raw_fastq"
output="${project_root}/results/quantification"
mkdir -p "${output}"

samples=(eipl_A eipl_C fikt_A fikt_C)
for sample in "${samples[@]}"; do
  "${salmon}" quant \
    --index "${index}" \
    --libType ISR \
    --mates1 "${fastq}/${sample}.1.fastq.gz" \
    --mates2 "${fastq}/${sample}.2.fastq.gz" \
    --threads 4 \
    --validateMappings \
    --seqBias \
    --gcBias \
    --output "${output}/${sample}"
done

