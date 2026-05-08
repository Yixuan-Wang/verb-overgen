#!/bin/bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=train-verb-overgen-trajectory
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

set -Eeuo pipefail
fail() {
  tyw tg text "#job fail name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "#job start name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"

python -m verb_overgen.train.train \
       trainer.max_epochs=3 \
       +trace.monitor_step.feature=causative
      #  +trace=voi_11

tyw tg text "#job finish name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} time=$(date '+%Y-%m-%d %H:%M:%S')"
