"""Run a testset on final checkpoints and save per-run v2-style result parquets."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import lightning as ptl
from loguru import logger
import polars as pl
import tokenizers as hftk
import torch
import wandb

from verb_overgen.common.tokenize import BOS_TOKEN, PAD_TOKEN
from verb_overgen.exp.collect_metadata import get_metadata
from verb_overgen.train.lib.dataloader_overgen import DatasetOvergen, TestOvergenOnceCallback, clean_dataset_overgen
from verb_overgen.train.lib.validate import last_sentence_index, load_unigram_prob_map, nll, slor
from verb_overgen.train.train import VerbOvergenTrainModule

WANDB_PATH = "tom-yx_wang/25-verb-overgen"
DEFAULT_TOKENIZER = Path("./data/tokenizer/1024-wordlevel/tokenizer.json")
ROOT_TRAIN = Path("var/train")


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


def load_unigram_index(unigram_dir: Path) -> dict[str, Path]:
    unigram_index: dict[str, Path] = {}
    with (unigram_dir / "metadata.jsonl").open() as f:
        for line in f:
            entry = json.loads(line)
            unigram_index[entry["internal"]["run_name"]] = unigram_dir / entry["path"]
    return unigram_index


def get_final_checkpoint_path(metadata: dict, *, root_train: Path = ROOT_TRAIN) -> Path:
    """Locate the final local checkpoint for a run.

    Prefer `last.ckpt` when present. Some older run directories only have
    epoch-named checkpoints, so fall back to the unique highest epoch.
    """
    run_name = metadata["internal"]["run_name"]
    last_matches = list(root_train.glob(f"*/{run_name}/checkpoints/last.ckpt"))
    if len(last_matches) == 1:
        return last_matches[0]
    if len(last_matches) > 1:
        raise FileNotFoundError(f"Expected 1 last.ckpt for {run_name!r}, found {len(last_matches)}: {last_matches}")

    epoch_matches: list[tuple[int, Path]] = []
    for path in root_train.glob(f"*/{run_name}/checkpoints/*epoch=*.ckpt"):
        match = re.search(r"epoch=(\d+)", path.name)
        if match is not None:
            epoch_matches.append((int(match.group(1)), path))

    if not epoch_matches:
        raise FileNotFoundError(f"No last.ckpt or epoch checkpoint found for {run_name!r}")

    max_epoch = max(epoch for epoch, _ in epoch_matches)
    max_epoch_paths = [path for epoch, path in epoch_matches if epoch == max_epoch]
    if len(max_epoch_paths) != 1:
        raise FileNotFoundError(
            f"Expected 1 checkpoint for max epoch={max_epoch:02d} for {run_name!r}, "
            f"found {len(max_epoch_paths)}: {max_epoch_paths}"
        )

    logger.info(f"[checkpoint fallback] {run_name}: using {max_epoch_paths[0].name}")
    return max_epoch_paths[0]


def load_testset_input(path_testset_input: Path, *, token_id_bos: int) -> pl.DataFrame:
    dataset = pl.read_parquet(path_testset_input)
    dataset = clean_dataset_overgen(dataset)
    assert dataset.select((pl.col("input_ids").list.get(0) == token_id_bos).all()).item(), (
        "All input_ids should start with BOS token"
    )
    return dataset


def run_testset_on_checkpoint(
    *,
    path_checkpoint: Path,
    dataset_test: DatasetOvergen,
    batch_size: int,
    accelerator: str,
    devices: str | int,
) -> pl.DataFrame:
    train_module = VerbOvergenTrainModule.load_from_checkpoint(path_checkpoint, map_location="cpu")
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_test.collate_fn,
    )

    test_callback = TestOvergenOnceCallback()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ptl.Trainer(
            callbacks=[test_callback],
            accelerator=accelerator,
            devices=devices,
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.test(train_module, dataloaders=dataloader_test)

    dataset_result = test_callback.test_dataset
    del train_module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dataset_result


def load_checkpoint_step_epoch(path_checkpoint: Path) -> tuple[int, int]:
    checkpoint = torch.load(path_checkpoint, map_location="meta", weights_only=False)
    step = int(checkpoint["global_step"])
    epoch = int(checkpoint.get("epoch", -1))
    return step, epoch


def prepare_result_dataset(
    *,
    dataset_result: pl.DataFrame,
    dataset_input: pl.DataFrame,
    metadata: dict,
    checkpoint_step: int,
    checkpoint_epoch: int,
    token_id_bos: int,
    unigram_path: Path | None,
) -> pl.DataFrame:
    dataset = dataset_result.with_columns(
        pl.lit(metadata["internal"]["run_name"]).alias("run_name"),
        pl.lit(metadata["seed"]).alias("run_seed"),
        pl.lit(metadata["manipulation"]).alias("run_manipulation"),
        pl.lit(metadata["scope"]).alias("run_scope"),
        pl.lit(metadata["target"]).alias("run_target"),
        pl.lit(metadata["target_verb"]).alias("run_target_verb"),
        pl.lit(checkpoint_step).alias("step"),
        pl.lit(checkpoint_epoch).alias("epoch"),
    )

    dataset = dataset.join(
        dataset_input.select(pl.all().exclude("verb", "kind")),
        on="index_sentence",
        how="left",
        coalesce=True,
        validate="m:1",
    )

    last_idx = last_sentence_index(pl.col("input_ids"), token_id_bos)
    slor_with_columns = []
    if unigram_path is not None:
        unigram_prob_map = load_unigram_prob_map(unigram_path)
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

    return dataset.select(
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
        *([pl.col("slor").alias("result_slor")] if unigram_path is not None else []),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument("--testset-input", type=Path, required=True, help="Parquet testset to evaluate")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-run result parquets")
    parser.add_argument("--unigram-dir", type=Path, default=None, help="Per-run unigram directory for SLOR")
    parser.add_argument("--no-slor", action="store_true", help="Skip SLOR computation and omit result_slor")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="Tokenizer JSON path")
    parser.add_argument("--batch-size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--accelerator", default="gpu", help="Lightning accelerator")
    parser.add_argument("--devices", default="1", help="Lightning devices argument")
    parser.add_argument("--append", action="store_true", help="Append new runs instead of replacing metadata.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of runs to save")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = hftk.Tokenizer.from_file(str(args.tokenizer))
    token_id_pad = tokenizer.token_to_id(PAD_TOKEN)
    token_id_bos = tokenizer.token_to_id(BOS_TOKEN)
    if token_id_pad is None or token_id_bos is None:
        raise ValueError(f"Tokenizer is missing required {PAD_TOKEN!r} or {BOS_TOKEN!r} token")

    compute_slor = not args.no_slor
    if compute_slor and args.unigram_dir is None:
        parser.error("--unigram-dir is required unless --no-slor is set")
    unigram_index = load_unigram_index(args.unigram_dir) if compute_slor else {}

    dataset_input = load_testset_input(args.testset_input, token_id_bos=token_id_bos)
    dataset_test = DatasetOvergen(dataframe=dataset_input, token_id_pad=token_id_pad)

    api = wandb.Api()
    runs = api.runs(WANDB_PATH, filters={"tags": {"$in": args.tags}})

    jsonl_path = output_dir / "metadata.jsonl"
    existing_run_ids, existing_paths = load_existing_metadata(jsonl_path) if args.append else (set(), set())
    jsonl_mode = "a" if args.append else "w"
    saved_count = 0

    with jsonl_path.open(jsonl_mode) as jsonl_file:
        for run in runs:
            if args.limit is not None and saved_count >= args.limit:
                break

            metadata = get_metadata(run, force_include_target_verb=True)
            if metadata is None:
                continue
            if metadata["target_verb"] == "glow":
                logger.warning(f"[skip] {metadata['internal']['run_name']}: glow has no causative usage")
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
                ckpt_path = get_final_checkpoint_path(metadata)
            except FileNotFoundError as e:
                logger.warning(f"[skip] {run_name}: {e}")
                continue

            if compute_slor and run_name not in unigram_index:
                raise FileNotFoundError(f"No unigram parquet found for run {run_name!r} in {args.unigram_dir}")

            logger.info(f"[eval] {run_name}: {ckpt_path}")
            dataset_result = run_testset_on_checkpoint(
                path_checkpoint=ckpt_path,
                dataset_test=dataset_test,
                batch_size=args.batch_size,
                accelerator=args.accelerator,
                devices=args.devices,
            )
            checkpoint_step, checkpoint_epoch = load_checkpoint_step_epoch(ckpt_path)
            dataset = prepare_result_dataset(
                dataset_result=dataset_result,
                dataset_input=dataset_input,
                metadata=metadata,
                checkpoint_step=checkpoint_step,
                checkpoint_epoch=checkpoint_epoch,
                token_id_bos=token_id_bos,
                unigram_path=unigram_index.get(run_name),
            )

            metadata_with_path = {**metadata, "path": str(rel_path)}
            dataset.write_parquet(save_path)
            jsonl_file.write(json.dumps(metadata_with_path) + "\n")
            if args.append:
                existing_run_ids.add(run_id)
                existing_paths.add(str(rel_path))
            saved_count += 1
            logger.info(f"[saved] {run_name} -> {save_path}")

    if saved_count == 0:
        raise RuntimeError("No testset results saved for any run.")

    logger.success(f"Saved {saved_count} run datasets to {output_dir}")


if __name__ == "__main__":
    main()
