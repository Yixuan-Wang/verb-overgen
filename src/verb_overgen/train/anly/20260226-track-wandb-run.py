import wandb
import os
import dotenv

import verb_overgen.common.startup  # noqa: F401

os.environ["WANDB_ENTITY"] = "tom-yx_wang"
os.environ["WANDB_PROJECT"] = "25-verb-overgen"

api = wandb.Api()
runs = api.runs("tom-yx_wang/25-verb-overgen", filters={"tags": {"$in": ["260226"]}})

subset_name = {
    "full": "O",
    "remove_preemption": "+(E-P)",
    "remove_entrenchment": "+P",
}

for run in runs:
    args = run.metadata.get("args", [])
    cmd = "-m verb_overgen.train.train " + " ".join(args)

    parsed = dict(a.split("=", 1) for a in args if "=" in a)
    seed = parsed.get("seed")
    shuffle_seed = parsed.get("data.train.shuffle_seed")
    subset_seed = parsed.get("data.train.subset_seed")
    subset = parsed.get("data.train.subset")

    print(f"{run.name}\t{run.state}\t({seed},{shuffle_seed},{subset_seed})\t{subset}\t{cmd}")

    # new_tags = [f"260226-seed-{seed}", f"260226-subset-{subset_name[subset]}"]
    # run.tags = [tag for tag in run.tags if tag != "full"] + new_tags
    # run.update()
