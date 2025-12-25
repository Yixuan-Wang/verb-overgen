#!/usr/bin/env bash

#SBATCH --account=aip-fredashi
#SBATCH --job-name=infer-scamp-con
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH -o logs/slurm/slurm-%A_%a.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a.err.txt

tyw tg text "start_job infer-scamp-con name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID}"

source ./.envrc
python -m verb_overgen.scamp_con \
  && tyw tg text "finish_job infer-scamp-con name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID}" \
  || tyw tg text "fail_job infer-scamp-con name=${SLURM_JOB_NAME} id=${SLURM_JOB_ID}"

