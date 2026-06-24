#! /bin/bash
#SBATCH --job-name=preprocess_tuev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/PAC-former/logs/%x-%j.log
#SBATCH --error=/scratch/zz5070/PAC-former/logs/%x-%j.log

source /scratch/zz5070/miniconda3/bin/activate
conda activate pacformer
cd /scratch/zz5070/PAC-former

mkdir -p logs
python scripts/preprocess_tuev.py
