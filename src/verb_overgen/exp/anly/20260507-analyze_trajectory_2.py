# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import os

    os.chdir(os.environ["PWD"])
    if "R_HOME" not in os.environ:
        import subprocess
        import json

        _direnv_env = subprocess.check_output(
            ["direnv", "export", "json"], text=True
        )
        if _direnv_env:
            os.environ.update(json.loads(_direnv_env))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import scipy.stats as stats

    from verb_overgen.util.r_arrow_bridge import RScript


@app.cell
def _():
    dataset_step = pl.scan_parquet("data/exp/trajectory/*.parquet")
    # dataset_step = dataset_step.with_columns(
    #     pl.when(pl.col("phase") == "pre").then(pl.col("step")).otherwise(pl.col("step") - 1).alias("step_effective")
    # )
    return (dataset_step,)


@app.cell
def _():
    dataset_testset = pl.read_parquet("./data/testset/output/testset_gemini.20260402T2045.parquet")
    dataset_testset_info = dataset_testset.select(
        "index_sentence",
        "verb",
        "index_verb_context",
        "sentence",
    )
    return (dataset_testset_info,)


@app.cell
def _(dataset_step):
    dataset_step_laugh: pl.DataFrame = (# type: ignore
        dataset_step.filter(pl.col("verb") == "laugh").collect()
    )
    dataset_step_laugh
    return (dataset_step_laugh,)


@app.cell
def _(dataset_step_laugh: pl.DataFrame, dataset_testset_info):
    dataset_step_diff_laugh = (
        dataset_step_laugh.group_by("step_effective", "index_sentence", "run_name")
        .agg(
            pl.col("kind").unique().item().alias("kind"),
            pl.col("target").unique().item().alias("target"),
            pl.col("run_seed").unique().item().alias("run_seed"),
            pl.col("feature_count").unique().item().alias("feature_count"),
            pl.col("nll_last_sentence").filter(pl.col("phase") == "pre").item().alias("nll_pre"),
            pl.col("nll_last_sentence").filter(pl.col("phase") == "post").item().alias("nll_post"),
        )
        .sort("step_effective", "index_sentence")
        .with_columns((pl.col("nll_post") - pl.col("nll_pre")).alias("nll_diff"))
        .join(dataset_testset_info, on="index_sentence",validate="m:1", coalesce=True)
    )
    dataset_step_diff_laugh
    return (dataset_step_diff_laugh,)


@app.cell
def _(dataset_step_diff_laugh):
    slider_index_verb_context = mo.ui.slider(
        **dataset_step_diff_laugh.select(pl.col("index_verb_context").min().alias("start"), pl.col("index_verb_context").max().alias("stop")).to_dicts()[0],
        show_value=True,
        debounce=True
    )
    slider_index_verb_context
    return (slider_index_verb_context,)


@app.cell
def _(slider_index_verb_context):
    INDEX_VERB_CONTEXT_TARGET = slider_index_verb_context.value
    return


@app.cell
def _(dataset_step_diff_laugh):
    dataset_step_diff_laugh
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First filter dataset_step_diff_laugh data with pl.col("kind") == "sentence_context_transitive".
    step_effective is the training step. index_verb_context is the index of the eval sentence.

    ```r
    nll_diff ~ log(step_effective) * data_kind + (1 + log(step_effective) | seed) + (1 | index_verb_context)
    ```
    """)
    return


@app.cell
def _():
    fit_trajectory = RScript("""
    library(arrow); library(dplyr); library(lme4); library(lmerTest); library(broom.mixed); library(emmeans)

    entry <- function(df) {
        df <- df %>%
            as_tibble() %>%
            mutate(target = relevel(factor(target), ref = "e"))

        model <- lmer(
            nll_diff ~ log(step_effective) * target
                + (1 + log(step_effective) | run_seed)
                + (1 | index_verb_context),
            data = df
        )

        fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
            mutate(table = "fixed")

        contrasts <- emmeans(model, pairwise ~ target)$contrasts %>%
            broom.mixed::tidy(conf.int = TRUE) %>%
            mutate(table = "contrasts")

        bind_rows(fixed, contrasts)
    }
    """)
    return (fit_trajectory,)


@app.cell
def _(dataset_step_diff_laugh, fit_trajectory):
    fit_trajectory_result = (
        dataset_step_diff_laugh
        .filter(
            pl.col("kind") == "sentence_context_transitive",
            pl.col("step_effective") > 0,
        )
        .select(
            "nll_diff",
            "step_effective",
            "target",
            "run_seed",
            "index_verb_context",
        )
        .pipe(fit_trajectory.call_table)
    )
    fit_trajectory_result
    return


@app.cell
def _(dataset_step_diff_laugh, fit_trajectory):
    fit_trajectory_causative_result = (
        dataset_step_diff_laugh
        .filter(
            pl.col("kind") == "sentence_context_causative",
            pl.col("step_effective") > 0,
        )
        .select(
            "nll_diff",
            "step_effective",
            "target",
            "run_seed",
            "index_verb_context",
        )
        .pipe(fit_trajectory.call_table)
    )
    fit_trajectory_causative_result
    return


@app.cell
def _(dataset_step_diff_laugh):
    _df = dataset_step_diff_laugh.filter(
        pl.col("index_verb_context") == 0,
        pl.col("step_effective") >= 1500,
        pl.col("run_seed") == 597,
        pl.col("kind") == "sentence_context_causative",
    )

    _df.group_by("target").agg(
        pl.col("nll_diff").mean().alias("nll_diff_mean"),
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
