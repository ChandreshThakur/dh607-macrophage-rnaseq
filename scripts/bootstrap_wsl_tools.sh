#!/usr/bin/env bash
set -euo pipefail

# User-space bootstrap for Windows hosts with WSL2; no sudo is required.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tools_root="${DH607_TOOLS_ROOT:-/mnt/c/tmp/dh607-tools}"
micromamba="${tools_root}/bin/micromamba"
environment="${DH607_ENV_PREFIX:-${HOME}/.local/share/dh607-env}"

mkdir -p "${tools_root}/bin"
if [[ ! -x "${micromamba}" ]]; then
  archive="${tools_root}/micromamba.tar.bz2"
  curl --fail --location --retry 3 \
    https://micro.mamba.pm/api/micromamba/linux-64/latest \
    --output "${archive}"
  # Python's stdlib handles bzip2 even on minimal WSL images lacking `bzip2`.
  python3 - "${archive}" "${tools_root}" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:bz2") as archive:
    archive.extract("bin/micromamba", path=sys.argv[2], filter="data")
PY
fi

"${micromamba}" create --yes --prefix "${environment}" --file "${project_root}/environment.yml"
"${micromamba}" run --prefix "${environment}" python --version
"${micromamba}" run --prefix "${environment}" fastqc --version
"${micromamba}" run --prefix "${environment}" multiqc --version
"${micromamba}" run --prefix "${environment}" salmon --version

printf 'Environment ready at %s\n' "${environment}"
