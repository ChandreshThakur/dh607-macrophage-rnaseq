#!/usr/bin/env bash
set -euo pipefail

# Extract Ubuntu's Salmon package and its non-base runtime libraries into the
# project-local ignored .tools directory. This avoids sudo/system modification.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deb_dir="${project_root}/.tools/salmon-debs"
runtime="${project_root}/.tools/salmon-runtime"
mkdir -p "${deb_dir}" "${runtime}"
cd "${deb_dir}"

apt-get download \
  salmon \
  libboost-filesystem1.90.0 \
  libboost-iostreams1.90.0 \
  libboost-program-options1.90.0 \
  libstaden-read14t64 \
  libhtscodecs2 \
  libtbb12 \
  libtbbmalloc2 \
  libtbbbind-2-5

for archive in ./*.deb; do
  dpkg-deb --extract "${archive}" "${runtime}"
done

export LD_LIBRARY_PATH="${runtime}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${runtime}/usr/bin/salmon" --version
ldd "${runtime}/usr/bin/salmon" | grep 'not found' && exit 1 || true

