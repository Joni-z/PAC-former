#! /bin/bash
#SBATCH --job-name=pacformer-pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/PAC-former/logs/%j.log
#SBATCH --error=/scratch/zz5070/PAC-former/logs/%j.log

# usage: sbatch pretrain.slurm pretrain_tuab_crossfreq
# Runs MAE pretraining + linear probe (pretrain.py) for the given config.

set -euo pipefail

# Slurm/cgroup exposes only the granted GPU as cuda:0; a stray
# CUDA_VISIBLE_DEVICES from the login shell can point at the wrong card.
unset CUDA_VISIBLE_DEVICES

# keep multiprocessing temp files off NFS home (avoids .nfs busy noise at exit)
export TMPDIR="${TMPDIR:-/scratch/zz5070/tmp}"
mkdir -p "$TMPDIR"

CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate pacformer
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /scratch/zz5070/PAC-former

CFG_NAME="$1"
CONFIG_PATH="configs/${CFG_NAME}.yaml"
mkdir -p logs

echo "Config: $CONFIG_PATH | Log: logs/${CFG_NAME}-${SLURM_JOB_ID}.log"

python -u pretrain.py --config "$CONFIG_PATH" 2>&1 | tee "logs/${CFG_NAME}-${SLURM_JOB_ID}.log"
