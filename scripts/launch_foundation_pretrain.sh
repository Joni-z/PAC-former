#!/usr/bin/env bash
set -euo pipefail

# Cluster-neutral launcher.  The scheduler allocation must expose the requested
# GPUs before this script runs (for Slurm, call it from an sbatch allocation).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/foundation/pacformer_base.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"

if [[ -z "${PACFORMER_DATA_ROOT:-}" ]]; then
  echo "PACFORMER_DATA_ROOT must point to the destination cluster's processed datasets" >&2
  exit 2
fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  foundation_pretrain.py --config "$CONFIG_PATH"
