import json
from pathlib import Path

import polars as pl
import wandb

from loguru import logger

from verb_overgen.exp.collect_metadata import get_metadata
from verb_overgen.exp.lib.unigram import count_unigram_from_dataloader
from verb_overgen.train.lib.utils import load_train_config
from verb_overgen.train.train import TrainingSession


def _filter_data_overrides(args: list[str]) -> list[str]:
    """Keep only seed and data-related overrides from a run's args list."""
    result = []
    for arg in args:
        if "=" not in arg:
            continue
        key, _ = arg.split("=", 1)
        bare_key = key.lstrip("+~")  # strip Hydra override prefixes (+, ++, ~)
        if bare_key == "seed" or bare_key == "data" or bare_key.startswith("data.train."):
            result.append(arg)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument("--output-dir", required=True, help="Directory to save unigram parquets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters={"tags": {"$in": args.tags}})

    jsonl_path = output_dir / "metadata.jsonl"
    with jsonl_path.open("w") as jsonl_file:
        for run in runs:
            metadata = get_metadata(run, force_include_target_verb=True)
            if metadata is None:
                continue

            run_name = metadata["internal"]["run_name"]

            if metadata["target_verb"] == "glow":
                logger.warning(f"[skip] {run_name}: `glow` results are discarded")
                continue

            run_args: list[str] = run.metadata.get("args", [])
            parsed_args = dict(a.split("=", 1) for a in run_args if "=" in a)
            shuffle_seed = parsed_args.get("data.train.shuffle_seed")
            subset_seed = parsed_args.get("data.train.subset_seed")

            metadata["internal"]["shuffle_seed"] = int(shuffle_seed) if shuffle_seed is not None else None
            metadata["internal"]["subset_seed"] = int(subset_seed) if subset_seed is not None else None

            overrides = _filter_data_overrides(run_args)
            logger.info(f"[info] {run_name} with overrides: {overrides}")

            try:
                cfg = load_train_config(overrides=overrides)
                session = TrainingSession(cfg)
                session.setup_tokenizer()
                session.setup_datasets()
                session.setup_dataloaders()
            except Exception as e:
                logger.warning(f"[skip] {run_name}: failed to build dataloader — {e}")
                continue

            df = count_unigram_from_dataloader(session.dataloader_train, progress=True)
            df = pl.select(
                pl.int_range(session.tokenizer.vocab_size).alias("token_id"),
            ).join(df, on="token_id", how="left").fill_null(0)

            manipulation = metadata["manipulation"]
            parts = [manipulation]
            if metadata["scope"]:
                parts.append(metadata["scope"])
            if metadata["target"]:
                parts.append(metadata["target"])
            if metadata["target_verb"]:
                parts.append(f"target-{metadata['target_verb']}")
            parts.append(f"seed-{metadata['seed']}")
            dataset_name = "_".join(parts)

            save_path = (output_dir / manipulation / dataset_name).with_suffix(".parquet")
            rel_path = save_path.relative_to(output_dir)
            metadata_with_path = {**metadata, "path": str(rel_path)}

            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(save_path)

            jsonl_file.write(json.dumps(metadata_with_path) + "\n")
            logger.info(f"[saved] {run_name} -> {save_path}")


if __name__ == "__main__":
    main()
