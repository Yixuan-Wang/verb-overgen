
# %%
from typing import Any
from itertools import starmap

import polars as pl
import numpy as np
import altair as alt

alt.data_transformers.enable("vegafusion")

# %%

# %%
run_metadata = {
    # "celestial-mountain-7": {
    #     "run_kind": "full",
    #     "run_seed": 42,

    #     "postprocess_step": True,
    # },
    # "wobbly-thunder-8": {
    #     "run_kind": "no-causative",
    #     "run_seed": 42,

    #     "postprocess_step": True,
    # },
    "swept-frost-12": {
        "run_kind": "no-causative",
        "run_seed": 37,
    },
    "hopeful-eon-16": {
        "run_kind": "no-causative",
        "run_seed": 23,
    },
    "rosy-sun-17": {
        "run_kind": "no-causative",
        "run_seed": 59,
    },
    "iconic-grass-19": {
        "run_kind": "full",
        "run_seed": 23,
    },
    "eager-bush-20": {
        "run_kind": "full",
        "run_seed": 37,
    },
    "whole-galaxy-21": {
        "run_kind": "full",
        "run_seed": 59,
    },
}

def ppl(token_probs):
    nlls = -np.log(token_probs)
    mean_nll = np.mean(nlls)
    ppl = np.exp(mean_nll)
    return ppl

def logprob(token_probs):
    return np.sum(np.log(token_probs))

def batch_ppl(token_probs_list):
    return pl.Series("ppl", [ppl(token_probs.to_numpy()) for token_probs in token_probs_list])

def batch_logprob(token_probs_list):
    return pl.Series("logprob", [logprob(token_probs.to_numpy()) for token_probs in token_probs_list])

# %%
def _(run_name: str, metadata: dict[str, Any]):
    df = pl.read_parquet(f"/home/tomyxw/projects/2025/verb-overgen/verb-overgen/var/train/20251230/{run_name}/test")

    if metadata.get("postprocess_step"):
        df = df.filter(pl.col("step") != 58731).with_columns(
            pl.when(pl.col("step") == 19576)
            .then(0)
            .when(pl.col("step") == 39153)
            .then(1)
            .when(pl.col("step") == 44048)
            .then(2)
            .when(pl.col("step") == 58730)
            .then(2)
            .alias("epoch")
        )

    df = df.with_columns(
        pl.lit(run_name).alias("run_name"),
        pl.lit(metadata["run_kind"]).alias("run_kind"),
        pl.lit(metadata["run_seed"]).alias("run_seed"),
    )

    return df

dataset_run = pl.concat(starmap(_, run_metadata.items()))
dataset_run.write_parquet("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/var/train/20251230/aggregate/testset_20260129.parquet")

# %%
def _(run_name: str, metadata: dict[str, Any]):
    df = pl.read_parquet(f"/home/tomyxw/projects/2025/verb-overgen/verb-overgen/var/train/20251230/{run_name}/eval")

    if metadata.get("postprocess_step"):
        df = df.filter(pl.col("step") < 58731).with_columns(
            pl.when(pl.col("step") <= 19576)
            .then(0)
            .when(pl.col("step") <= 39153)
            .then(1)
            .when(pl.col("step") <= 58730)
            .then(2)
            .alias("epoch")
        )

    df = df.with_columns(
        pl.lit(run_name).alias("run_name"),
        pl.lit(metadata["run_kind"]).alias("run_kind"),
        pl.lit(metadata["run_seed"]).alias("run_seed"),
    )

    return df

dataset_eval = pl.concat(starmap(_, run_metadata.items()))
dataset_eval.write_parquet("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/var/train/20251230/aggregate/eval_scamp_20260129.parquet")
