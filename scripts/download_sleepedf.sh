#!/bin/bash
# Download Sleep-EDF Cassette (SC* files) from PhysioNet.
# Uses SHA256SUMS to build exact file list (avoids HTML parsing issues).
# No account required -- data is publicly licensed (ODbL).
#
# Usage:
#   bash scripts/download_sleepedf.sh [raw_dir]
#   default raw_dir: sleep_edf/raw

set -euo pipefail

RAW_DIR="${1:-/scratch/zz5070/PAC-former/sleep_edf/raw}"
BASE="https://physionet.org/files/sleep-edfx/1.0.0"

mkdir -p "$RAW_DIR"

echo "==> Fetching file list from SHA256SUMS..."
wget -q "${BASE}/SHA256SUMS.txt" -O "${RAW_DIR}/.sha256sums.txt"

# Extract only SC Cassette files
grep "sleep-cassette/SC" "${RAW_DIR}/.sha256sums.txt" \
    | awk '{print $2}' \
    | sed 's|sleep-cassette/||' \
    > "${RAW_DIR}/.filelist.txt"

total=$(wc -l < "${RAW_DIR}/.filelist.txt")
echo "==> Found ${total} SC files to download (~1.7 GB total)"

cd "$RAW_DIR"
count=0
while IFS= read -r fname; do
    if [[ -f "$fname" ]]; then
        count=$((count + 1))
        continue
    fi
    wget -q -c "${BASE}/sleep-cassette/${fname}" -O "${fname}"
    count=$((count + 1))
    if (( count % 20 == 0 )); then
        echo "  ${count}/${total} done..."
    fi
done < .filelist.txt

echo "Done. ${total} files in ${RAW_DIR}"
