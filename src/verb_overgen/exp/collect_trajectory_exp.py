import json
from pathlib import Path

import wandb
import polars as pl

from loguru import logger

from verb_overgen.exp.collect_metadata import get_metadata

ROOT_TRAIN = "var/train"


def get_monitor_paths_per_verb(metadata) -> dict[str, Path]:
    """Return {verb: path} for all per-verb monitor parquets found for this run."""
    run_name = metadata["internal"]["run_name"]
    matches = list(Path(ROOT_TRAIN).glob(f"*/{run_name}/eval/monitor_step_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No monitor_step_*.parquet found for run {run_name!r}")
    return {path.stem.removeprefix("monitor_step_"): path for path in matches}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted dataset")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters={"tags": {"$in": args.tags}})

    jsonl_path = output_dir / "metadata.jsonl"
    with jsonl_path.open("w") as jsonl_file:
        for run in runs:
            metadata = get_metadata(run)
            if metadata is None:
                continue

            run_name = metadata["internal"]["run_name"]
            try:
                verb_paths = get_monitor_paths_per_verb(metadata)
            except FileNotFoundError as e:
                logger.warning(f"[skip] {run_name}: {e}")
                continue

            for verb, dataset_path in verb_paths.items():
                dataset = pl.scan_parquet(dataset_path)
                dataset = dataset.select(
                    pl.lit(run_name).alias("run_name"),
                    pl.lit(metadata["seed"]).alias("run_seed"),
                    pl.all(),
                    pl.when(pl.col("phase") == "pre").then(pl.col("step")).otherwise(pl.col("step") - 1).alias("step_effective"),
                )

                dataset_name = f"{verb}_seed-{metadata['seed']}"
                save_path = (output_dir / dataset_name).with_suffix(".parquet")
                rel_path = save_path.relative_to(output_dir)
                metadata_with_path = {**metadata, "verb": verb, "path": str(rel_path)}

                save_path.parent.mkdir(parents=True, exist_ok=True)
                dataset.sink_parquet(save_path)

                jsonl_file.write(json.dumps(metadata_with_path) + "\n")
                logger.info(f"[saved] {run_name}/{verb} -> {save_path}")


if __name__ == "__main__":
    main()
