#! /bin/bash
#SBATCH --job-name=consolidate_chbmit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:30:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_63_general

#SBATCH --output=/scratch/zz5070/PAC-former/logs/%x-%j.log
#SBATCH --error=/scratch/zz5070/PAC-former/logs/%x-%j.log

set -euo pipefail
CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate pacformer
cd /scratch/zz5070/PAC-former

python -u scripts/consolidate_pkl_dataset.py --format tuab \
    --processed_dir chb_mit/processed --splits train val test
