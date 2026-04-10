"""
Use Wandb SDK to query all runs tagged with 260402 that is in state "finished", and fetch the following metrics:

"train/verb_feature/this/{causative,causative_entrenched}"
"eval/loss_mean/this/{causative,transitive,context_causative,context_transitive}"
"""

from __future__ import annotations

import os
from pathlib import Path

import ipdb
import polars as pl
import wandb
from loguru import logger
from omegaconf import OmegaConf

import verb_overgen.common.startup as _startup

os.environ["WANDB_ENTITY"] = "tom-yx_wang"
os.environ["WANDB_PROJECT"] = "25-verb-overgen"

METRIC_KEYS = [
    "train/verb_feature/this/causative",
    "train/verb_feature/this/causative_entrenched",
    "eval/loss_mean/this/causative",
    "eval/loss_mean/this/transitive",
    "eval/loss_mean/this/context_causative",
    "eval/loss_mean/this/context_transitive",
]


@ipdb.iex
def collect(sweep_filters) -> pl.DataFrame:
    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters=sweep_filters)

    all_rows: list[dict] = []

    for run in runs:
        cfg = OmegaConf.create(run.config)

        run_name = run.name
        run_subset = cfg.data_config.train.subset
        run_verb_of_interest = cfg.data_config.train.get("subset_verb_of_interest", None)

        if "seed" not in cfg:
            run_seed = next(int(arg.lstrip("seed=")) for arg in run.metadata["args"] if arg.startswith("seed="))
        else:
            run_seed = cfg.seed

        history = run.scan_history()
        len_history = 0
        for row in history:
            len_history += 1
            record = {
                "run_name": run_name,
                "run_subset": run_subset,
                "verb": run_verb_of_interest,
                "run_seed": run_seed,
                "step": row.get("trainer/global_step"),
            }
            for key in METRIC_KEYS:
                record[key] = row.get(key)
            all_rows.append(record)
        logger.info(
            f"Fetching {len_history} history records for run: {run_name} (seed={run_seed}, subset={run_subset}, verb_of_interest={run_verb_of_interest})"
        )

    if not all_rows:
        raise RuntimeError("No trajectory data found for any run.")

    dataset = pl.DataFrame(
        all_rows,
        schema={
            "run_name": pl.String,
            "run_subset": pl.String,
            "verb": pl.String,
            "run_seed": pl.Int64,
            "step": pl.Int64,
            "train/verb_feature/this/causative": pl.Int64,
            "train/verb_feature/this/causative_entrenched": pl.Int64,
            "eval/loss_mean/this/causative": pl.Float64,
            "eval/loss_mean/this/transitive": pl.Float64,
            "eval/loss_mean/this/context_causative": pl.Float64,
            "eval/loss_mean/this/context_transitive": pl.Float64,
        },
    )
    dataset = dataset.filter(pl.all_horizontal(pl.selectors.starts_with("eval").is_not_null()))

    return dataset


def main() -> None:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output parquet path (default: ./data/exp/overgen_trajectory-<tags>-<timestamp>.parquet)",
    )
    args = parser.parse_args()

    sweep_filters = {"tags": {"$in": args.tags}, "state": "finished"}

    if args.output is None:
        tags_str = "_".join(args.tags)
        timestamp = datetime.now().strftime("%y%m%dT%H%M")
        output_file = f"./data/exp/overgen_trajectory-{tags_str}-{timestamp}.parquet"
    else:
        output_file = args.output

    if Path(output_file).exists():
        raise FileExistsError(f"Output file already exists: {output_file}")

    logger.info(f"Collecting trajectories for runs with filters: {sweep_filters}")
    dataset = collect(sweep_filters)

    __import__("ipdb").set_trace()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(output_file)
    logger.success(f"Trajectory data collected and saved to {output_file}")


if __name__ == "__main__":
    main()
