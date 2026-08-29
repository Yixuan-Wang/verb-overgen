"""Collect test results across all runs in a wandb sweep tag and save per-run parquets."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from loguru import logger
import wandb

from verb_overgen.exp.collect_metadata import get_metadata
from verb_overgen.train.lib.dataloader_overgen import clean_dataset_overgen
from verb_overgen.train.lib.validate import (
    last_sentence_index,
    load_unigram_prob_map,
    nll,
    slor,
)

ROOT_TRAIN = "var/train"

TESTSET_INPUT_PATH = Path("./data/testset/output/testset_template_0818_wordlevel_pp.parquet")
TOKEN_ID_BOS = 1


def load_testset_input() -> pl.DataFrame:
    dataset = pl.read_parquet(TESTSET_INPUT_PATH)
    dataset = clean_dataset_overgen(dataset)
    assert dataset.select(
        (pl.col("input_ids").list.get(0) == TOKEN_ID_BOS).all()
    ).item(), "All input_ids should start with BOS token"
    return dataset


def get_test_data_path(run_name: str) -> list[Path]:
    matches = list(Path(ROOT_TRAIN).glob(f"*/{run_name}/test/overgen_step_*.parquet"))
    matches = [p for p in matches if p.stem != "overgen_step_00_00000000"]
    if not matches:
        raise FileNotFoundError(f"No test parquets found for {run_name!r}")
    return matches


def get_dataset_name(metadata: dict) -> str:
    parts = [metadata["manipulation"]]
    if metadata["scope"]:
        parts.append(metadata["scope"])
    if metadata["target"]:
        parts.append(metadata["target"])
    if metadata["target_verb"]:
        parts.append(f"target-{metadata['target_verb']}")
    parts.append(f"seed-{metadata['seed']}")
    return "_".join(parts)


def load_existing_metadata(jsonl_path: Path) -> tuple[set[str], set[str]]:
    existing_run_ids = set()
    existing_paths = set()
    if not jsonl_path.exists():
        return existing_run_ids, existing_paths

    with jsonl_path.open() as f:
        for line in f:
            entry = json.loads(line)
            existing_run_ids.add(entry["internal"]["run_id_wandb"])
            existing_paths.add(entry["path"])
    return existing_run_ids, existing_paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted datasets")
    parser.add_argument("--unigram-dir", required=False, default=None, help="Directory containing per-run unigram parquets and metadata.jsonl (from collect_unigram.py). Required unless --no-slor is set.")
    parser.add_argument("--no-slor", action="store_true", help="Skip SLOR computation and omit result_slor from output. Does not require --unigram-dir.")
    parser.add_argument("--append", action="store_true", help="Append new runs to an existing output directory instead of replacing metadata.jsonl.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_slor = not args.no_slor
    if compute_slor and args.unigram_dir is None:
        parser.error("--unigram-dir is required unless --no-slor is set")

    unigram_index: dict[str, Path] = {}
    if compute_slor:
        unigram_dir = Path(args.unigram_dir)
        with (unigram_dir / "metadata.jsonl").open() as f:
            for line in f:
                entry = json.loads(line)
                unigram_index[entry["internal"]["run_name"]] = unigram_dir / entry["path"]

    dataset_input = load_testset_input()

    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters={"tags": {"$in": args.tags}})

    jsonl_path = output_dir / "metadata.jsonl"
    existing_run_ids, existing_paths = load_existing_metadata(jsonl_path) if args.append else (set(), set())
    jsonl_mode = "a" if args.append else "w"
    with jsonl_path.open(jsonl_mode) as jsonl_file:
        for run in runs:
            metadata = get_metadata(run, force_include_target_verb=True)
            if metadata is None:
                continue

            if metadata["target_verb"] == "glow":
                logger.warning(f"[skip] {metadata['internal']['run_name']}: skip glow, which has no causative usage in the training set.")
                continue

            run_name = metadata["internal"]["run_name"]
            run_id = metadata["internal"]["run_id_wandb"]
            if args.append and run_id in existing_run_ids:
                logger.info(f"[skip] {run_name}: already collected")
                continue

            dataset_name = get_dataset_name(metadata)
            save_path = (output_dir / dataset_name).with_suffix(".parquet")
            rel_path = save_path.relative_to(output_dir)
            if args.append and str(rel_path) in existing_paths:
                raise FileExistsError(f"Refusing to overwrite existing parquet path in append mode: {rel_path}")
            if args.append and save_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing parquet file in append mode: {save_path}")

            try:
                test_paths = get_test_data_path(run_name)
            except FileNotFoundError as e:
                logger.warning(f"[skip] {run_name}: {e}")
                continue

            dataset = pl.read_parquet(test_paths)
            dataset = dataset.with_columns(
                pl.lit(run_name).alias("run_name"),
                pl.lit(metadata["seed"]).alias("run_seed"),
                pl.lit(metadata["manipulation"]).alias("run_manipulation"),
                pl.lit(metadata["scope"]).alias("run_scope"),
                pl.lit(metadata["target"]).alias("run_target"),
                pl.lit(metadata["target_verb"]).alias("run_target_verb"),
            )

            dataset = dataset.join(
                dataset_input.select(pl.all().exclude("verb", "kind")),
                on="index_sentence",
                how="left",
                coalesce=True,
                validate="m:1",
            )
            last_idx = last_sentence_index(pl.col("input_ids"), TOKEN_ID_BOS)
            slor_with_columns = []
            if compute_slor:
                unigram_prob_map = load_unigram_prob_map(unigram_index[run_name])
                slor_with_columns = [
                    slor(
                        token_prob=pl.col("token_prob").list.slice(last_idx),
                        input_ids=pl.col("input_ids").list.slice(last_idx + 1),
                        unigram_prob_map=unigram_prob_map,
                    ).alias("slor"),
                ]
            dataset = dataset.with_columns(
                pl.col("token_prob").list.slice(last_idx).pipe(nll).alias("nll"),
                *slor_with_columns,
            )
            dataset = dataset.select(
                pl.col("run_name"),
                pl.col("run_seed"),
                pl.col("run_manipulation"),
                pl.col("run_scope"),
                pl.col("run_target"),
                pl.col("run_target_verb"),
                pl.col("step").alias("ckpt_step"),
                pl.col("epoch").alias("ckpt_epoch"),
                pl.col("verb").alias("eval_verb"),
                pl.col("index_sentence").alias("eval_index_sentence"),
                pl.col("index_verb_context").alias("eval_index_verb_context"),
                pl.col("kind").str.contains("context").alias("eval_has_context"),
                pl.col("kind").str.extract(r"(causative|transitive)").alias("eval_construction"),
                pl.col("token_prob").alias("result_token_prob"),
                pl.col("nll").alias("result_nll"),
                *([pl.col("slor").alias("result_slor")] if compute_slor else []),
            )

            metadata_with_path = {**metadata, "path": str(rel_path)}

            save_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.write_parquet(save_path)

            jsonl_file.write(json.dumps(metadata_with_path) + "\n")
            if args.append:
                existing_run_ids.add(run_id)
                existing_paths.add(str(rel_path))
            logger.info(f"[saved] {run_name} -> {save_path}")


if __name__ == "__main__":
    main()
