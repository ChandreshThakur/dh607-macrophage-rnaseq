#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${project_root}/.tools/salmon-runtime"
if [[ ! -x "${runtime}/usr/bin/salmon" ]]; then
  exec salmon "$@"
fi
export LD_LIBRARY_PATH="${runtime}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${runtime}/usr/bin/salmon" "$@"

