#!/bin/bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=exp-collect-unigram
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=4
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

set -Eeuo pipefail

declare -a CONFIGS=(
    "260402"
    "260402-full"
    "260410"
    "260506-except-verb"
    "260507-substitute"
)

# Select config based on SLURM array task ID
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"


fail() {
  tyw tg text "#job fail task=${CONFIG} name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "#job start task=${CONFIG} name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"

OUTPUT_DIR="var/results/v2/stats-unigram"
mkdir -p "$OUTPUT_DIR/$CONFIG"

uv run -m verb_overgen.exp.collect_unigram \
        "$CONFIG" \
        --output-dir "$OUTPUT_DIR/$CONFIG"

tyw tg text "#job finish task=${CONFIG} name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-0} time=$(date '+%Y-%m-%d %H:%M:%S')"
