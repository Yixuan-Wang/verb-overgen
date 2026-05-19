"""Print run configs for all runs in a wandb sweep tag."""

from __future__ import annotations

import os
from typing import Final

from loguru import logger
import wandb
from omegaconf import OmegaConf

os.environ["WANDB_ENTITY"] = "tom-yx_wang"
os.environ["WANDB_PROJECT"] = "25-verb-overgen"

SCOPES_WITH_TARGET_VERB: Final[set[str]] = {"per-verb", "except-verb", "per-verb-no-abstract"}

MAP_INTERNAL_SUBSET_TO_CONDITION: Final[dict[str, dict[str, str]]] = {
    "full": {
        "manipulation": "none",
        "scope": "",
        "target": "",
    },
    "remove_preemption": {
        "manipulation": "remove",
        "scope": "all-verb",
        "target": "p"
    },
    "remove_entrenchment": {
        "manipulation": "remove",
        "scope": "all-verb",
        "target": "e",
    },
    "remove_preemption_on_verb": {
        "manipulation": "remove",
        "scope": "per-verb",
        "target": "p"
    },
    "remove_entrenchment_on_verb": {
        "manipulation": "remove",
        "scope": "per-verb",
        "target": "e",
    },
    "remove_preemption_except_verb": {
        "manipulation": "remove",
        "scope": "except-verb",
        "target": "p"
    },
    "remove_entrenchment_except_verb": {
        "manipulation": "remove",
        "scope": "except-verb",
        "target": "e",
    },
    "remove_entrenchment_on_verb_no_abstract": {
        "manipulation": "remove",
        "scope": "per-verb-no-abstract",
        "target": "e",
    },
    "substitute_preemption_on_verb": {
        "manipulation": "substitute",
        "scope": "per-verb",
        "target": "p", # Notice that e is removed besides p-substitute.
    },
}

def get_metadata(
    run,
    *,
    force_include_target_verb: bool = False,
):
    cfg = OmegaConf.create(run.config)

    metadata_run_name = run.name
    metadata_run_id_wandb = run.id
    metadata_tags: list[str] = run.tags
    metadata_subset =  cfg.data_config.train.subset

    if metadata_subset not in MAP_INTERNAL_SUBSET_TO_CONDITION:
        logger.warning(f"Skip run {metadata_run_name}: unknown subset {metadata_subset}")
        return None

    if "seed" not in cfg:
        if "args" not in run.metadata:
            __import__("ipdb").set_trace()

        metadata_seed = next(int(arg.lstrip("seed=")) for arg in run.metadata["args"] if arg.startswith("seed="))
    else:
        metadata_seed = cfg.seed

    if force_include_target_verb or MAP_INTERNAL_SUBSET_TO_CONDITION[metadata_subset]["scope"] in SCOPES_WITH_TARGET_VERB:
        metadata_target_verb: str = cfg.data_config.train.subset_verb_of_interest
    else:
        metadata_target_verb = ""

    logger.info(f"Retrieved {metadata_run_name}")

    return {
        **MAP_INTERNAL_SUBSET_TO_CONDITION[metadata_subset],
        "target_verb": metadata_target_verb,
        "seed": metadata_seed,
        "internal": {
            "run_name": metadata_run_name,
            "run_id_wandb": metadata_run_id_wandb,
            "run_subset": metadata_subset,
            "run_tags": metadata_tags,
        }
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="WandB tags to filter runs by (matched with $in)")
    parser.add_argument("--force-include-target-verb", action="store_true", help="Whether to include the target verb in the metadata regardless of the manipulation scope.")
    args = parser.parse_args()

    sweep_filters = {"tags": {"$in": args.tags}}

    api = wandb.Api()
    runs = api.runs("tom-yx_wang/25-verb-overgen", filters=sweep_filters)

    list_metadata = []
    for run in runs:
        metadata = get_metadata(run, force_include_target_verb=args.force_include_target_verb)
        if not metadata:
            continue
        list_metadata.append(metadata)

    import json
    with open("metadata.json", "w") as fp:
        json.dump(list_metadata, fp, indent=4)


if __name__ == "__main__":
    main()
