#!/usr/bin/env bash
set -euo pipefail

# Build clean mmap arrays for the unlabeled pool.  TUSZ/CHB-MIT downstream
# preprocessing contains dense "_add" seizure windows created using labels.
# Those remain valid for supervised finetuning, but are excluded here so the
# self-supervised corpus does not receive label-derived sampling information or
# thousands of near-duplicate overlapping windows.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${PACFORMER_DATA_ROOT:-}" ]]; then
  echo "PACFORMER_DATA_ROOT must point to the processed dataset root" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/consolidate_pkl_dataset.py \
  --format tuab \
  --processed_dir "${PACFORMER_DATA_ROOT}/tusz/processed" \
  --splits train \
  --out_names ssl_train \
  --exclude_substring _add

"$PYTHON_BIN" scripts/consolidate_pkl_dataset.py \
  --format tuab \
  --processed_dir "${PACFORMER_DATA_ROOT}/chbmit/processed" \
  --splits train \
  --out_names ssl_train \
  --exclude_substring _add
