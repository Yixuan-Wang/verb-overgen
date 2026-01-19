#!/bin/bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=train-verb-overgen
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o logs/slurm/slurm-%j.out.txt
#SBATCH -e logs/slurm/slurm-%j.err.txt

set -Eeuo pipefail
fail() {
  tyw tg text "fail_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "start_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"
python -m verb_overgen.train.train

tyw tg text "finish_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"
