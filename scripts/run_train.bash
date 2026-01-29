#!/bin/bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=packing-full-20-epochs
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=37
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

set -Eeuo pipefail
fail() {
  tyw tg text "fail_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "start_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"
python -m verb_overgen.train.train \
       seed=${SLURM_ARRAY_TASK_ID} \
       data.train.shuffle_seed=${SLURM_ARRAY_TASK_ID} \
       trainer.max_epochs=20 \
       trainer.log_every_n_steps=500

tyw tg text "finish_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"
