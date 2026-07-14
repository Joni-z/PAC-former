#!/bin/bash
# Download CHB-MIT Scalp EEG Database from PhysioNet.
# Uses SHA256SUMS to build exact file list (avoids HTML parsing issues).
# No account required -- data is publicly licensed (ODbL).
# Downloads in parallel (xargs -P) since a single sequential wget is slow
# against PhysioNet's per-connection rate limit.
#
# Usage:
#   bash scripts/download_chbmit.sh [raw_dir] [parallelism]
#   default raw_dir: chb_mit/raw, default parallelism: 10

set -euo pipefail

RAW_DIR="${1:-/scratch/zz5070/PAC-former/chb_mit/raw}"
JOBS="${2:-10}"
BASE="https://physionet.org/files/chbmit/1.0.0"

mkdir -p "$RAW_DIR"

echo "==> Fetching file list from SHA256SUMS..."
wget -q "${BASE}/SHA256SUMS.txt" -O "${RAW_DIR}/.sha256sums.txt"

# Extract only SC Cassette-equivalent files (all top-level chbNN/* entries)
grep "chb" "${RAW_DIR}/.sha256sums.txt" \
    | awk '{print $2}' \
    > "${RAW_DIR}/.filelist.txt"

total=$(wc -l < "${RAW_DIR}/.filelist.txt")
echo "==> Found ${total} files to download (~42 GB total), ${JOBS}-way parallel"

cd "$RAW_DIR"
mkdir -p $(awk -F/ '{print $1}' .filelist.txt | sort -u)

fetch_one() {
    fname="$1"
    # -s (not just -f): repeated interrupted runs left 0-byte files behind,
    # and a plain existence check treated those as "already downloaded" --
    # they'd never get retried. Require non-empty too.
    [[ -s "$fname" ]] && return 0
    wget -q -c "${BASE}/${fname}" -O "${fname}"
}
export -f fetch_one
export BASE

xargs -a .filelist.txt -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {}

echo "Done. ${total} files in ${RAW_DIR}"
