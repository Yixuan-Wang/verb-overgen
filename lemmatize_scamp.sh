#!/usr/bin/env bash

#SBATCH --partition=COMPLING
#SBATCH --job-name=verb-overgen-lemmatize-scamp
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o logs/slurm-%j.out.txt # File to which STDOUT will be written
#SBATCH -e logs/slurm-%j.err.txt # File to which STDERR will be written

cd "$HOME/projects/2025/verb-overgen/repo"
source .venv/bin/activate

export RAY_DATA_VERBOSE_PROGRESS=0

python -m verb_overgen.prep.lemmatize_scamp \
    && tyw tg text "Lemmatize finished" \
    || tyw tg text "Lemmatize failed"