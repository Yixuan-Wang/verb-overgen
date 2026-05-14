#!/bin/bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=exp-collect-testset-v2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH -o logs/slurm/slurm-%j_%x.out.txt
#SBATCH -e logs/slurm/slurm-%j_%x.err.txt

set -Eeuo pipefail

TAGS=(
    "260402"
    "260402-full"
    "260410"
    "260506-except-verb"
    "260507-substitute"
)

fail() {
  tyw tg text "#job fail name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "#job start name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"

OUTPUT_DIR="var/results/v2/testset"
UNIGRAM_DIR="var/results/v2/stats-unigram"
mkdir -p "$OUTPUT_DIR"

uv run -m verb_overgen.exp.collect_testset_v2 \
        "${TAGS[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --unigram-dir "$UNIGRAM_DIR"

tyw tg text "#job finish name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"
