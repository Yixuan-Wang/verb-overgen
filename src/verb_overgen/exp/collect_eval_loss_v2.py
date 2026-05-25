"""Collect overgeneralization eval losses from wandb runs and save per-run parquets.

For each run, this extracts:
- eval/loss_mean/{verb}/causative
- eval/loss_mean/{verb}/transitive

Each parquet has one row per run/step/verb/construction loss value.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf
import polars as pl
import wandb

from verb_overgen.common.features import VERB_TARGET_FINALE
from verb_overgen.exp.collect_metadata import get_metadata

WANDB_ENTITY = os.environ.setdefault("WANDB_ENTITY", "tom-yx_wang")
WANDB_PROJECT = os.environ.setdefault("WANDB_PROJECT", "25-verb-overgen")
WANDB_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

CONSTRUCTIONS = ("causative", "transitive")
DEFAULT_VERBS = VERB_TARGET_FINALE


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


def collect_run_dataset(run, metadata: dict, *, override_verbs: Sequence[str] | None = None) -> pl.DataFrame | None:
    if metadata["target_verb"]:
        verbs = [metadata["target_verb"]]
    else:
        verbs = DEFAULT_VERBS

    run_name = metadata["internal"]["run_name"]
    # keyed by (step, verb); values accumulate per-construction losses
    step_verb_data: dict[tuple[Any, str], dict[str, float | None]] = {}
    history_len = 0

    for history_row in run.scan_history():
        history_len += 1
        step = history_row.get("trainer/global_step")

        for verb in verbs:
            key = (step, verb)
            if key not in step_verb_data:
                step_verb_data[key] = {"loss_causative": None, "loss_transitive": None}
            for construction in CONSTRUCTIONS:
                loss = history_row.get(f"eval/loss_mean/{verb}/{construction}")
                if loss is not None:
                    step_verb_data[key][f"loss_{construction}"] = float(loss)

    rows: list[dict[str, Any]] = [
        {
            "run_name": run_name,
            "run_subset": metadata["internal"]["run_subset"],
            "run_seed": metadata["seed"],
            "run_manipulation": metadata["manipulation"],
            "run_scope": metadata["scope"],
            "run_target": metadata["target"],
            "run_target_verb": metadata["target_verb"],
            "step": step,
            "eval_verb": verb,
            "eval_has_context": False,
            **losses,
        }
        for (step, verb), losses in step_verb_data.items()
        if losses["loss_causative"] is not None or losses["loss_transitive"] is not None
    ]

    if rows:
        logger.info(f"[collected] {run_name}: {len(rows)} loss rows from {history_len} history rows")
    else:
        logger.warning(f"[skip] {run_name}: no requested eval loss metrics in {history_len} history rows")
        return None

    return pl.DataFrame(
        rows,
        schema_overrides={
            "run_name": pl.String,
            "run_subset": pl.String,
            "run_seed": pl.Int64,
            "run_manipulation": pl.String,
            "run_scope": pl.String,
            "run_target": pl.String,
            "run_target_verb": pl.String,
            "step": pl.Int64,
            "eval_verb": pl.String,
            "eval_has_context": pl.Boolean,
            "loss_causative": pl.Float64,
            "loss_transitive": pl.Float64,
        },
    ).sort(["run_name", "step", "eval_verb"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save per-run eval loss parquets and metadata.jsonl",
    )
    parser.add_argument(
        "--verbs",
        nargs="+",
        default=None,
        help="Optional explicit verbs to collect. Defaults to trace.tokens_to_watch from each run config.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new runs to an existing output directory instead of replacing metadata.jsonl.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(WANDB_PATH, filters={"tags": {"$in": args.tags}, "state": "finished"})

    jsonl_path = output_dir / "metadata.jsonl"
    existing_run_ids, existing_paths = load_existing_metadata(jsonl_path) if args.append else (set(), set())
    jsonl_mode = "a" if args.append else "w"
    saved_count = 0

    logger.info(f"Collecting eval losses for finished runs tagged with: {args.tags}")
    with jsonl_path.open(jsonl_mode) as jsonl_file:
        for run in runs:
            metadata = get_metadata(run)
            if metadata is None:
                continue

            run_name = metadata["internal"]["run_name"]
            run_id = metadata["internal"]["run_id_wandb"]
            if args.append and run_id in existing_run_ids:
                logger.info(f"[skip] {run_name}: already collected")
                continue

            dataset_name = get_dataset_name(metadata)
            save_path = (output_dir / metadata["manipulation"] / dataset_name).with_suffix(".parquet")
            rel_path = save_path.relative_to(output_dir)
            if args.append and str(rel_path) in existing_paths:
                raise FileExistsError(f"Refusing to overwrite existing parquet path in append mode: {rel_path}")
            if args.append and save_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing parquet file in append mode: {save_path}")

            dataset = collect_run_dataset(run, metadata, override_verbs=args.verbs)
            if dataset is None:
                continue

            metadata_with_path = {**metadata, "path": str(rel_path)}

            save_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.write_parquet(save_path)

            jsonl_file.write(json.dumps(metadata_with_path) + "\n")
            if args.append:
                existing_run_ids.add(run_id)
                existing_paths.add(str(rel_path))
            saved_count += 1
            logger.info(f"[saved] {run_name} -> {save_path}")

    if saved_count == 0:
        raise RuntimeError("No eval loss data saved for any run.")

    logger.success(f"Saved {saved_count} run datasets to {output_dir}")


if __name__ == "__main__":
    main()
