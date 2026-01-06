#!/usr/bin/env bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=testset-gen
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

set -Eeuo pipefail
fail() {
  tyw tg text "fail_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "start_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"
python -m verb_overgen.prep.testset_gen

tyw tg text "finish_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"
