"""Collect test results across all runs in a wandb sweep tag and save to a single parquet."""

from __future__ import annotations
from IPython.core.inputtransformer2 import tr

import os
from pathlib import Path

import polars as pl
from loguru import logger
import wandb
from omegaconf import OmegaConf

from verb_overgen.train.lib.dataloader_overgen import clean_dataset_overgen
from verb_overgen.train.lib.validate import nll, VERB_TARGET_17, VERB_TARGET_5

os.environ["WANDB_ENTITY"] = "tom-yx_wang"
os.environ["WANDB_PROJECT"] = "25-verb-overgen"

TESTSET_INPUT_PATH = Path("./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet")
# OUTPUT_DIR = Path("./var/exp")


def load_testset_input() -> pl.DataFrame:
    dataset = pl.read_parquet(TESTSET_INPUT_PATH)
    dataset = clean_dataset_overgen(dataset)
    return dataset


def load_run_test_data(train_dir: Path, run_name: str, run_meta: dict) -> pl.DataFrame:
    test_dir = train_dir / run_name / "test"
    if not test_dir.exists():
        raise ValueError(f"Test directory does not exist for run: {run_name}")
        
    df = pl.read_parquet(test_dir).with_columns(
        **{k: pl.lit(v) for k, v in run_meta.items()}
    )

    return df

@__import__("ipdb").iex
def collect(sweep_filters, train_dir: Path) -> pl.DataFrame:
    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters=sweep_filters)

    dataset_input = load_testset_input()

    all_dfs: list[pl.DataFrame] = []
    for run in runs:
        cfg = OmegaConf.create(run.config)
        
        run_name = run.name
        run_flavor = cfg.data_config.train.subset
        
        if "seed" not in cfg:
            run_seed = next(int(arg.lstrip("seed=")) for arg in run.metadata["args"] if arg.startswith("seed="))
        else:
            run_seed = cfg.seed
        
        run_meta = {
            "run_name": run_name,
            "run_flavor": run_flavor,
            "run_seed": run_seed,
        }

        df = load_run_test_data(train_dir, run.name, run_meta)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No test data found for any run.")

    dataset = pl.concat(all_dfs)
    dataset = dataset.filter(pl.col("step") != 0)
    dataset = dataset.with_columns(pl.col("token_prob").pipe(nll)).rename({"token_prob": "nll"})
    dataset = dataset.join(dataset_input.select(pl.all().exclude("verb", "kind")), on="index_sentence", coalesce=True)

    return dataset


def main() -> None:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument(
        "-t", "--train-dir",
        default="./var/train/20260402",
        help="Training output directory (default: ./var/train/20260402)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output parquet path (default: ./data/exp/overgen_results-raw-<tags>-<timestamp>.parquet)",
    )
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    sweep_filters = {"tags": {"$in": args.tags}}

    if args.output is None:
        tags_str = "_".join(args.tags)
        timestamp = datetime.now().strftime("%y%m%dT%H%M")
        output_file = f"./data/exp/overgen_results-raw-{tags_str}-{timestamp}.parquet"
    else:
        output_file = args.output

    if Path(output_file).exists():
        raise FileExistsError(f"Output file already exists: {output_file}")

    logger.info(f"Collecting test results for runs with filters: {sweep_filters}")
    dataset = collect(sweep_filters, train_dir)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(output_file)
    logger.success(f"Test results collected and saved to {output_file}")

if __name__ == "__main__":
    main()
