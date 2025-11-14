#!/bin/bash

#SBATCH --partition=COMPLING
#SBATCH --job-name=verb-overgen-train
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-%j.out.txt # File to which STDOUT will be written
#SBATCH -e logs/slurm-%j.err.txt # File to which STDERR will be written
 
cd "$HOME/projects/2025/verb-overgen/repo"
uv run -m verb_overgen.train.train \
    && tyw tg text "Finished sbatch job $SLURM_JOB_ID" \
    || tyw tg text "Failed sbatch job $SLURM_JOB_ID"