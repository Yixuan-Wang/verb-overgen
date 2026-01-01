#!/usr/bin/env bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=infer-scamp-con
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH -o logs/slurm/slurm-%A_%a.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a.err.txt

set -Eeuo pipefail
fail() {
  tyw tg text "fail_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "start_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"
python -m verb_overgen.scamp_con.infer_scamp

tyw tg text "finish_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"
