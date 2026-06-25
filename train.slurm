#! /bin/bash
#SBATCH --job-name=pacformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint='h200'
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/PAC-former/logs/%x-%j.log
#SBATCH --error=/scratch/zz5070/PAC-former/logs/%x-%j.log

# usage: sbatch train.slurm configs/tuab_mi.yaml

source /scratch/zz5070/miniconda3/etc/profile.d/conda.sh
conda activate pacformer
cd /scratch/zz5070/PAC-former

mkdir -p logs
python -u train.py --config "$1"
