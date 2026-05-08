# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import scipy.stats as stats

    from verb_overgen.util.r_arrow_bridge import RScript

    os.chdir(os.environ["PWD"])


@app.cell
def _():
    dataset_step = pl.scan_parquet("./var/train/20260402/scarlet-wind-554/eval/monitor_step_causative.parquet")
    dataset_step = dataset_step.with_columns(
        pl.when(pl.col("phase") == "pre").then(pl.col("step")).otherwise(pl.col("step") - 1).alias("step_aligned")
    )
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
        dataset_step_laugh.group_by("step_aligned", "index_sentence")
        .agg(
            pl.col("kind").unique().item().alias("kind"),
            pl.col("feature_count").unique().item().alias("feature_count"),
            pl.col("nll_last_sentence").filter(pl.col("phase") == "pre").item().alias("nll_pre"),
            pl.col("nll_last_sentence").filter(pl.col("phase") == "post").item().alias("nll_post"),
        )
        .sort("step_aligned", "index_sentence")
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
    return (INDEX_VERB_CONTEXT_TARGET,)


@app.cell
def _(INDEX_VERB_CONTEXT_TARGET, dataset_step_diff_laugh):
    _df = (
        dataset_step_diff_laugh
            .filter(
                pl.col("index_verb_context") == INDEX_VERB_CONTEXT_TARGET,
                pl.col("kind").str.starts_with("sentence_context")
            )
            # .group_by("kind")
            # .agg(
            #     pl.col("step_aligned"),
            #     pl.col("nll_diff").cum_sum()
            # )
            # .explode("step_aligned", "nll_diff")
    )

    _fig, _ax = plt.subplots(figsize=(16, 4))

    _ax = sns.lineplot(
        _df,
        x="step_aligned",
        y="nll_diff",
        hue="kind",
        ax=_ax,
    )

    _ax.set_ylim(-0.25, 0.25)

    _ax
    return


@app.cell
def _(INDEX_VERB_CONTEXT_TARGET, dataset_step_diff_laugh):
    _fig, _ax = plt.subplots(figsize=(16, 4))

    _df = (
        dataset_step_diff_laugh.filter(
            pl.col("kind").str.contains("context"),
            pl.col("index_verb_context") == INDEX_VERB_CONTEXT_TARGET,
        )
        .group_by("step_aligned")
        .agg(
            pl.col("nll_pre").filter(pl.col("kind") == "sentence_context_causative").item().alias("nll_pre_causative"),
            pl.col("nll_pre").filter(pl.col("kind") == "sentence_context_transitive").item().alias("nll_pre_transitive"),
        )
        .with_columns((pl.col("nll_pre_causative") - pl.col("nll_pre_transitive")).alias("nll_pre_gap"))
    )

    _ax = sns.lineplot(
        _df,
        x="step_aligned",
        y="nll_pre_gap",
        ax=_ax,
    )

    _ax.axhline(0, color="gray", linestyle="--")

    _ax
    return


@app.cell
def _(dataset_step_diff_laugh):
    _fig, _ax = plt.subplots(figsize=(16, 4))

    _ax = sns.lineplot(
        dataset_step_diff_laugh.filter(
            pl.col("kind").str.contains("context"),
        ).with_columns(
            pl.col("kind").str.strip_prefix("sentence_context_")
        ),
        x="step_aligned",
        y="nll_pre",
        hue="kind",
        ax=_ax,
        errorbar="ci",
        # palette="Paired"
    )

    _ax.set_ylim(2, 9)
    _ax.get_legend().set_title("Eval Construction Type")
    _ax.set_xlabel("Training Step")
    _ax.set_ylabel("Eval Loss")

    _ax
    return


@app.cell
def _(INDEX_VERB_CONTEXT_TARGET, dataset_step_diff_laugh):
    _fig, _ax = plt.subplots(figsize=(16, 4))

    _ax = sns.lineplot(
        dataset_step_diff_laugh.filter(
            pl.col("kind").str.contains("context"),
            pl.col("index_verb_context") == INDEX_VERB_CONTEXT_TARGET,
        ),
        x="step_aligned",
        y="nll_pre",
        hue="kind",
        ax=_ax,
        # errorbar="ci",
        # palette="Paired"
    )

    _ax.set_ylim(4, 9)

    _ax
    return


@app.cell
def _(INDEX_VERB_CONTEXT_TARGET, dataset_step_diff_laugh):
    _df = (
        dataset_step_diff_laugh
            .filter(
                pl.col("index_verb_context") == INDEX_VERB_CONTEXT_TARGET,
            )
            .group_by("step_aligned")
            .agg(
                pl.col("nll_diff").filter(pl.col("kind") == "sentence_context_causative").item().alias("nll_diff_causative"),
                pl.col("nll_diff").filter(pl.col("kind") == "sentence_context_transitive").item().alias("nll_diff_transitive"),
                pl.col("feature_count").unique().item()
            )
            .with_columns(
                pl.col("step_aligned").pipe(lambda s: (s >= 00) & (s <= 4000)).alias("is_step_1500_3000")
            )
    )

    _ax = sns.scatterplot(
        _df,
        x="nll_diff_causative",
        y="nll_diff_transitive",
        hue="is_step_1500_3000",
    )

    # y = x
    _ax.plot([-0.1, 0.1], [-0.1, 0.1], color="gray", linestyle="--")

    _pearson_r = stats.pearsonr(
        _df["nll_diff_causative"].to_numpy(), _df["nll_diff_transitive"].to_numpy()
    )

    mo.vstack([
        _pearson_r,
        _ax,
    ])
    return


@app.cell
def _(dataset_step_diff_laugh):
    _df = (
        dataset_step_diff_laugh
            .group_by("index_verb_context", "step_aligned")
            .agg(
                pl.col("nll_diff").filter(pl.col("kind") == "sentence_context_causative").item().alias("nll_diff_causative"),
                pl.col("nll_diff").filter(pl.col("kind") == "sentence_context_transitive").item().alias("nll_diff_transitive"),
                pl.col("feature_count").unique().item()
            )
            .sort("index_verb_context", "step_aligned")
            .unpivot(
                on=["nll_diff_causative", "nll_diff_transitive"],
                index=["index_verb_context", "step_aligned", "feature_count"],
                value_name="nll_diff",
                variable_name="kind",
            )
            .with_columns(
                pl.col("kind").str.strip_prefix("nll_diff_"),
            )
    )

    _fig, _ax = plt.subplots(figsize=(16, 4))

    _ax = sns.lineplot(
        _df,
        x="step_aligned",
        y="nll_diff",
        hue="kind",
        ax=_ax,
    )

    _ax.set_ylim(-0.15, 0.15)
    _ax.get_legend().set_title("Eval Construction Type")
    _ax.set_xlabel("Training Step")
    _ax.set_ylabel("Eval Loss Reduction")

    _ax
    return


@app.cell
def _(dataset_step_diff_laugh):
    _df = (
        dataset_step_diff_laugh
            .group_by("index_verb_context", "step_aligned")
            .agg(
                pl.col("nll_diff").filter(pl.col("kind") == "sentence_context_causative").item().alias("nll_diff_causative"),
                pl.col("nll_diff").filter(pl.col("kind") == "sentence_context_transitive").item().alias("nll_diff_transitive"),
                pl.col("feature_count").unique().item()
            )
            .sort("index_verb_context", "step_aligned")
            .with_columns(
                pl.col("step_aligned").floordiv(1912).alias("step_aligned_group")
            )
    )

    _grid = (
        sns.FacetGrid(
        _df,
        col="step_aligned_group",
        col_wrap=4,
    )
        .map_dataframe(
            sns.regplot,
            x="nll_diff_causative",
            y="nll_diff_transitive",
            marker=".",
            line_kws={"color": "red"},
        )
    )

    for _idx, _ax in enumerate(_grid.axes.flatten()):
        _ax.set_aspect("equal")
        _ax.plot([-0.2, 0.15], [-0.2, 0.15], color="gray", linestyle="--")
        _ax.set_title(f"epoch = {_idx * 0.5}")

    mo.vstack([
        stats.pearsonr(
            _df["nll_diff_causative"].to_numpy(), _df["nll_diff_transitive"].to_numpy(),
        ),
        _grid,
    ])
    return


@app.cell
def _(dataset_step_diff_laugh):
    dataset_step_gap_laugh = (
        dataset_step_diff_laugh
        .filter(pl.col("kind").str.contains("context"))
        .group_by("step_aligned", "index_verb_context")
        .agg(
            pl.col("nll_pre").filter(pl.col("kind") == "sentence_context_causative").mean().alias("nll_pre_causative"),
            pl.col("nll_pre").filter(pl.col("kind") == "sentence_context_transitive").mean().alias("nll_pre_transitive"),
        )
        .with_columns((pl.col("nll_pre_causative") - pl.col("nll_pre_transitive")).alias("nll_pre_gap"))
        .sort("index_verb_context", "step_aligned")
    )
    dataset_step_gap_laugh
    return


@app.cell
def _(dataset_step_diff_laugh):
    dataset_step_diff_laugh.group_by("index_verb_context", "kind").agg(
        pl.col("nll_diff").filter(
            pl.col("step_aligned") >= 1500,
            pl.col("step_aligned") <= 3000,
        ).std().alias("nll_diff_2000_std")
    )
    return


if __name__ == "__main__":
    app.run()
