#!/bin/bash
#SBATCH --job-name=pac-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --output=/scratch/zz5070/PAC-former/logs/%j.log
#SBATCH --error=/scratch/zz5070/PAC-former/logs/%j.log

# CPU-only diagnostics (no --gres=gpu, so these do NOT consume the GPU QOS
# budget the evidence jobs are queued against).
# usage: sbatch cpu.slurm scripts/band_locality.py sleepedf tuab
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export TMPDIR="${TMPDIR:-/scratch/zz5070/tmp}"
mkdir -p "$TMPDIR"
source /scratch/zz5070/miniconda3/etc/profile.d/conda.sh
conda activate pacformer
cd /scratch/zz5070/PAC-former
TAG=$(basename "$1" .py)
python -u "$@" 2>&1 | tee "logs/${TAG}-${SLURM_JOB_ID}.log"
