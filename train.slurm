#! /bin/bash
#SBATCH --job-name=pacformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/PAC-former/logs/%j.log
#SBATCH --error=/scratch/zz5070/PAC-former/logs/%j.log

# usage: sbatch train.slurm pacint_tuev_measured
#        sbatch --time=10:00:00 --mem=32G train.slurm ours_scratch_tuab_bandidx
#
# ---------------------------------------------------------------------------
# Resource defaults are sized for the LIGHT profile (TUEV / Sleep-EDF
# supervised), measured from `sacct` over 172 completed train.slurm jobs:
#
#   profile  datasets              max walltime   max MaxRSS
#   light    tuev                  1.15 h         20.2 GB
#            sleep                 2.68 h          5.8 GB
#   heavy    tuab                  6.84 h         32.0 GB  <- pressed the limit
#            tusz                  7.78 h         32.0 GB  <- pressed the limit
#            chbmit                9.29 h         32.0 GB  <- pressed the limit
#
# Heavy datasets ran flush against the old 32 GB cap and one earlier job was
# killed OUT_OF_MEMORY, so the cap is NOT slack for them -- it is the working
# set. They must raise --mem/--time on the sbatch command line; the guard below
# refuses to start rather than let a heavy run die at the 3 h mark with a
# half-finished log. CLI flags override the #SBATCH lines above.
#
# CPU: the old 16 cores ran at 8.78% efficiency (job 14937685: 37 min of CPU
# over 27 min wall = ~1.4 cores busy). 8 matches `num_workers: 8` in the configs
# so every loader worker still owns a core.
#
# NOTE: the usual pending reason here is QOSGrpGRES -- a cap on concurrent GPUs
# for the account. Trimming CPU/RAM does not lift that; only the shorter --time
# helps, by making the job eligible for backfill into scheduling gaps.
# ---------------------------------------------------------------------------

set -euo pipefail

# Slurm/cgroup 通常只把分到的那块卡曝露成 cuda:0；登录 shell 带过来的
# CUDA_VISIBLE_DEVICES 可能指错卡，和 SLURM 自己的可见性设置混用会报 No CUDA GPUs。
unset CUDA_VISIBLE_DEVICES

cd /scratch/zz5070/PAC-former

CFG_NAME="$1"
CONFIG_PATH="configs/${CFG_NAME}.yaml"
mkdir -p logs

[ -f "$CONFIG_PATH" ] || { echo "no such config: $CONFIG_PATH" >&2; exit 1; }

# --- resource guard -------------------------------------------------------
# Runs before conda/python so it costs seconds, not a wasted allocation. Reads
# the dataset from the YAML and compares it against what Slurm actually gave us.
DATASET=$(sed -n 's/^dataset:[[:space:]]*//p' "$CONFIG_PATH" | head -1 | tr -d '"'"'"' \r')
case "$DATASET" in
    tuab|tusz|chbmit) NEED_MIN=600; NEED_MB=32000 ;;   # heavy: >=10 h, >=32 GB
    tuev)             NEED_MIN=180; NEED_MB=24000 ;;   # light
    sleep*)           NEED_MIN=240; NEED_MB=16000 ;;
    *)                NEED_MIN=0;   NEED_MB=0     ;;   # unknown -> trust the user
esac

# Both probes FAIL OPEN: if Slurm does not report a value we let the run start.
# A guard that blocks every job because an env var was renamed is worse than the
# timeout it prevents. Only a value we actually read, and that is genuinely too
# small, stops the run.
GOT_MIN=""
TL=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^TimeLimit=//p' || true)
if [[ "$TL" =~ ^[0-9]+(-[0-9]+)?(:[0-9]+)+$ ]]; then          # [DD-]HH:MM:SS
    GOT_MIN=$(awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else print $1*60+$2 }' <<<"$TL")
fi
GOT_MB="${SLURM_MEM_PER_NODE:-}"

SHORT=""
if [ -n "$GOT_MIN" ] && [ "$GOT_MIN" -lt "$NEED_MIN" ]; then
    SHORT="${SHORT}  time: got ${GOT_MIN} min, '$DATASET' needs ${NEED_MIN}"$'\n'
fi
if [ -n "$GOT_MB" ] && [ "$GOT_MB" -lt "$NEED_MB" ]; then
    SHORT="${SHORT}  mem : got ${GOT_MB} MB, '$DATASET' needs ${NEED_MB}"$'\n'
fi
if [ -n "$SHORT" ]; then
    cat >&2 <<EOF
=============================================================================
REFUSING TO START -- allocation is below what '$DATASET' has historically used.
${SHORT}Resubmit with:
  sbatch --time=$((NEED_MIN/60)):00:00 --mem=$((NEED_MB/1000))G train.slurm ${CFG_NAME}
Failing now instead of timing out mid-run with an unusable partial log.
=============================================================================
EOF
    exit 1
fi
echo "Resources OK for '$DATASET': ${GOT_MIN:-?} min, ${GOT_MB:-?} MB, ${SLURM_CPUS_PER_TASK:-?} cpus"
# --------------------------------------------------------------------------

CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate pacformer
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "Config: $CONFIG_PATH | Log: logs/${CFG_NAME}-${SLURM_JOB_ID}.log"

# %j.log above (slurm's own --output) and this tee'd, config-named log both
# get the full output -- so you keep a job-id-keyed copy AND one named after
# the run for easy diffing across the mi/attention/cotar ablation.
python -u train.py --config "$CONFIG_PATH" 2>&1 | tee "logs/${CFG_NAME}-${SLURM_JOB_ID}.log"
