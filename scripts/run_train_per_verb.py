#!/usr/bin/env -S uv run --no-project --script
#SBATCH --account=aip-fredashi
#SBATCH --job-name=verb-overgen-260402
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --array=0-109%22
#SBATCH --time=3:00:00
#SBATCH -o logs/slurm/slurm-%A_%a_%x.out.txt
#SBATCH -e logs/slurm/slurm-%A_%a_%x.err.txt

# /// script
# dependencies = ["sh", "numpy"]
# ///

import json
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import sh

VERBS_OF_INTEREST = [
    "go", "sneeze", "blink", "look", "rise", "glow",
    "laugh", "sing", "sleep", "smile", "cry",
]

SUBSETS = [
    # "data.train.subset=full",
    "data.train.subset=remove_preemption_on_verb",
    "data.train.subset=remove_entrenchment_on_verb",
]

# 3 seed triples (seed, shuffle_seed, subset_seed), each in range(0, 1024)
_rng_old = np.random.default_rng(3407)
_rng = np.random.default_rng(1337)
SEEDS: list[tuple[int, int, int]] = [
    tuple(row) for row in _rng_old.integers(0, 1024, size=(3, 3)).tolist()
] + [
    tuple(row) for row in _rng.integers(0, 1024, size=(2, 3)).tolist()
]


def notify(event: str, line: int | None = None):
    job_name = os.environ.get("SLURM_JOB_NAME", "")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{event} name={job_name} id={job_id} array={array_id}"
    if line is not None:
        msg += f" line={line}"
    msg += f" time={time_str}"
    sh.tyw("tg", "text", msg, _fg=True)


def load_direnv():
    env_json = sh.direnv("export", "json")
    if env_json:
        os.environ.update(json.loads(str(env_json)))


def main():
    notify("start_job")

    load_direnv()

    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    seed_idx, verb_idx, subset_idx = np.unravel_index(
        array_id, (len(SEEDS), len(VERBS_OF_INTEREST), len(SUBSETS))
    )
    verb = VERBS_OF_INTEREST[verb_idx]
    seed, shuffle_seed, subset_seed = SEEDS[seed_idx]
    subset = SUBSETS[subset_idx]

    sh.uv(
        "run", "-m", "verb_overgen.train.train",
        "trainer.max_epochs=3",
        "wandb.tags=['260402']",
        f"seed={seed}",
        f"data.train.shuffle_seed={shuffle_seed}",
        f"data.train.subset_seed={subset_seed}",
        subset,
        f"data.train.subset_verb_of_interest={verb}",
        _fg=True,
    )

    notify("finish_job")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _, _, tb = sys.exc_info()
        lineno = traceback.extract_tb(tb)[-1].lineno
        notify("fail_job", line=lineno)
        raise
