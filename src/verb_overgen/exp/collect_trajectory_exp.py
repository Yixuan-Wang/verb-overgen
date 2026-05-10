import regex as re
import json
from pathlib import Path

import torch
import transformers as hf
import tokenizers as hftk
import wandb
import polars as pl

from loguru import logger

from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.common.tokenize import BOS_TOKEN, UNK_TOKEN, PAD_TOKEN
from verb_overgen.exp.collect_metadata import get_metadata

ROOT_TRAIN = "var/train"

def get_monitor_data(metadata) -> Path:
    run_name = metadata["internal"]["run_name"]
    matches = list(Path(ROOT_TRAIN).glob(f"*/{run_name}/eval/monitor_step*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 dataset for {run_name!r}, found {len(matches)}: {matches}")
    return matches[0]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted dataset")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer_raw = hftk.Tokenizer.from_file(args.tokenizer)
    # tokenizer = hf.PreTrainedTokenizerFast(
    #     tokenizer_object=tokenizer_raw,
    #     unk_token=UNK_TOKEN,
    #     pad_token=PAD_TOKEN,
    #     bos_token=BOS_TOKEN,
    #     eos_token=BOS_TOKEN,
    # )
    # tokenizer.save_pretrained(output_dir)

    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters={"tags": {"$in": args.tags}})

    jsonl_path = output_dir / "metadata.jsonl"
    with jsonl_path.open("w") as jsonl_file:
        for run in runs:
            metadata = get_metadata(run, force_include_target_verb=True)
            if metadata is None:
                continue

            metadata["manipulation"] = "moment"
            monitored_feature: str = next(filter(None, (re.search(r"trace\.monitor_step\.feature=([A-Za-z0-9_-]+)", arg) for arg in run.metadata["args"]))).group(1)
            if monitored_feature == FEATURE_CAUSATIVE:
                metadata["target"] = "p"
            elif monitored_feature == FEATURE_CAUSATIVE_ENTRENCHED:
                metadata["target"] = "e"
            else:
                logger.warning(f"Skip run {metadata['internal']['run_name']}: unknown monitored feature {monitored_feature}")
                continue

            run_name = metadata["internal"]["run_name"]
            try:
                dataset_path = get_monitor_data(metadata)
            except FileNotFoundError as e:
                logger.warning(f"[skip] {run_name}: {e}")
                continue

            dataset = pl.scan_parquet(dataset_path)
            dataset = dataset.select(
                pl.lit(run_name).alias("run_name"),
                pl.lit(metadata["seed"]).alias("run_seed"),
                pl.lit(metadata["target"]).alias("target"),
                pl.all(),
                pl.when(pl.col("phase") == "pre").then(pl.col("step")).otherwise(pl.col("step") - 1).alias("step_effective"),
            )

        
            

            parts = []
            if metadata["target"]:
                parts.append(metadata["target"])
            if metadata["target_verb"]:
                parts.append(f"target-{metadata['target_verb']}")
            parts.append(f"seed-{metadata['seed']}")
            # parts.append("last")
            dataset_name = "_".join(parts)

            save_path = (output_dir / dataset_name).with_suffix(".parquet")
            rel_path = save_path.relative_to(output_dir)
            metadata_with_path = {**metadata, "path": str(rel_path)}
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.sink_parquet(save_path)

            jsonl_file.write(json.dumps(metadata_with_path) + "\n")
            logger.info(f"[saved] {run_name} -> {save_path}")


if __name__ == "__main__":
    main()
