#! /bin/bash
#SBATCH --job-name=pacformer
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

# usage: sbatch train.slurm tuab_mi

set -euo pipefail

# Slurm/cgroup 通常只把分到的那块卡曝露成 cuda:0；登录 shell 带过来的
# CUDA_VISIBLE_DEVICES 可能指错卡，和 SLURM 自己的可见性设置混用会报 No CUDA GPUs。
unset CUDA_VISIBLE_DEVICES

CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate pacformer
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /scratch/zz5070/PAC-former

CFG_NAME="$1"
CONFIG_PATH="configs/${CFG_NAME}.yaml"
mkdir -p logs

echo "Config: $CONFIG_PATH | Log: logs/${CFG_NAME}-${SLURM_JOB_ID}.log"

# %j.log above (slurm's own --output) and this tee'd, config-named log both
# get the full output -- so you keep a job-id-keyed copy AND one named after
# the run for easy diffing across the mi/attention/cotar ablation.
python -u train.py --config "$CONFIG_PATH" 2>&1 | tee "logs/${CFG_NAME}-${SLURM_JOB_ID}.log"
