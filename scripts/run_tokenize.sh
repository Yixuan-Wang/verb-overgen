#!/bin/bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=tokenize-verb-overgen
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

set -Eeuo pipefail
fail() {
  tyw tg text "fail_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} line=$1 time=$(date '+%Y-%m-%d %H:%M:%S')"
}
trap 'fail $LINENO' ERR

tyw tg text "start_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"

eval "$(direnv export bash)"

# Run

<<train-pychildes
python -m verb_overgen.train.run_tokenize \
  --tokenizer-file "./data/tokenizer/1024-wordlevel/tokenizer.json" \
  --input-path "./data/pychildes-morphosyntax.20251024.parquet" \
  --is-pychildes \
  --output-path "./data/pychildes-morphosyntax-tokenized.20251024.parquet" \
  --cpu ${SLURM_CPUS_ON_NODE}
train-pychildes

python -m verb_overgen.train.run_tokenize \
  --tokenizer-file "./data/tokenizer/1024-wordlevel/tokenizer.json" \
  --input-path "./data/scamp/scamp-postprocessed.parquet" \
  --output-path "./data/scamp/scamp-postprocessed-tokenized.20260101.parquet" \
  --cpu ${SLURM_CPUS_ON_NODE}

tyw tg text "finish_job name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} time=$(date '+%Y-%m-%d %H:%M:%S')"
