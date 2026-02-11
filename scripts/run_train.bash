#!/bin/bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=packing-debug
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

set -Eeuo pipefail
fail() {
  tyw tg text "fail_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "start_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"

# declare -a CONFIGS=(
#     ""
# )

# # Select config based on SLURM array task ID
# CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

python -m verb_overgen.train.train \
       trainer.max_epochs=3 

tyw tg text "finish_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} time=$(date '+%Y-%m-%d %H:%M:%S')"
