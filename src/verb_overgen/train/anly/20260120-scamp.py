# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.19.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from functools import partial
    from typing import Literal

    import apfel
    import marimo as mo
    import polars as pl
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import altair as alt

    alt.data_transformers.enable("vegafusion")
    os.chdir(os.environ["PWD"])

    # set font family
    plt.rcParams["font.family"] = "JetBrains Mono"
    return pl, plt, sns


@app.cell
def _():
    from apfel import not_none
    return


@app.cell
def _(pl):
    def ppl(token_probs: pl.Expr):
        # nlls = -np.log(token_probs)
        nlls = token_probs.list.eval(pl.element().log().neg())
        # mean_nll = np.mean(nlls)
        mean_nll = nlls.list.mean()
        # ppl = np.exp(mean_nll)
        ppl = mean_nll.exp()
        return ppl
    return (ppl,)


@app.cell
def _(pl):
    dataset_input = pl.read_parquet("./data/scamp/scamp-postprocessed-tokenized.20260101.parquet").with_row_index("index_sentence")
    return (dataset_input,)


@app.cell
def _(pl, ppl):
    dataset_run = pl.scan_parquet("./var/train/20251230/aggregate/eval_scamp_20260123.parquet")

    dataset_run = dataset_run.select(
        pl.all().exclude("token_prob"),
        pl.col("token_prob").pipe(ppl).alias("ppl"),
        pl.concat_str(
            pl.col("run_kind"),
            pl.lit("-"),
            pl.col("run_seed").cast(pl.String),
        ).alias("run")
    )
    return (dataset_run,)


@app.cell
def _(dataset_input):
    dataset_input
    return


@app.cell
def _(dataset_input, pl):
    dataset_input_core = dataset_input.select(
        pl.col("index_sentence"),
        pl.col("file_name"),
        pl.col("file_line_idx"),
        pl.col("is_grammatical"),
        pl.col("is_lemma_ok"),
        pl.col("sentence_raw")
    ).join(
        dataset_input.filter(pl.col("is_llm_verdict").is_not_null()).select("file_name", "file_line_idx", "is_llm_verdict"),
        on=["file_name", "file_line_idx"],
        how="left"
    ).with_columns(
        pl.col("is_llm_verdict").fill_null(False)
    )
    return (dataset_input_core,)


@app.cell
def _(dataset_input_core, dataset_run, pl):
    dataset_one = dataset_run.join(
        dataset_input_core.lazy(),
        on="index_sentence",
        how="left"
    )

    ppl_grammatical = pl.col("ppl").filter(pl.col("is_grammatical")).item()
    ppl_ungrammatical = pl.col("ppl").filter(~pl.col("is_grammatical")).item()

    dataset_one = dataset_one.with_row_index("_row_idx").group_by("step", "run", "file_name", "file_line_idx").agg(
        pl.col("_row_idx", "is_lemma_ok", "is_llm_verdict").first(),
        ppl_grammatical.alias("ppl_grammatical"),
        ppl_ungrammatical.alias("ppl_ungrammatical"),
        (ppl_ungrammatical > ppl_grammatical).alias("correct"),
    ).sort("_row_idx").select(pl.all().exclude("_row_idx")).collect()
    return (dataset_one,)


@app.cell
def _(dataset_one, pl, sns):
    _df = dataset_one.group_by("step", "run").agg(
        pl.col("correct").filter(pl.col("is_lemma_ok"), pl.col("is_llm_verdict").not_()).mean().alias("accuracy_-conceivable"),
        pl.col("correct").filter(pl.col("is_lemma_ok")).mean().alias("accuracy_full"),
        pl.col("correct").filter(pl.col("is_llm_verdict"), pl.col("is_lemma_ok")).mean().alias("accuracy_+conceivable"),
    ).sort("step").unpivot(
        on=pl.selectors.starts_with("accuracy"),
        index=["step", "run"],
        variable_name="subset",
        value_name="accuracy",
    ).with_columns(
        pl.col("subset").str.replace("accuracy_", ""),
    )

    sns.lineplot(
        _df,
        x="step",
        y="accuracy",
        hue="subset",
    )
    return


@app.cell
def _(dataset_one, pl, sns):
    _df = dataset_one.filter(pl.col("is_lemma_ok")).group_by("step", "file_name", "file_line_idx").agg(
        pl.col("ppl_grammatical", "ppl_ungrammatical").log().std().name.replace("ppl_", "std_nll_"),
        pl.col("is_lemma_ok", "is_llm_verdict").first(),
    ).sort("step", "file_name", "file_line_idx").group_by("step").agg(
        pl.col("std_nll_grammatical").filter("is_llm_verdict").mean().alias("mean_std_nll_+gram_+con"),
        pl.col("std_nll_grammatical").filter(pl.col("is_llm_verdict").not_()).mean().alias("mean_std_nll_+gram_-con"),
        pl.col("std_nll_ungrammatical").filter("is_llm_verdict").mean().alias("mean_std_nll_-gram_+con"),
        pl.col("std_nll_ungrammatical").filter(pl.col("is_llm_verdict").not_()).mean().alias("mean_std_nll_-gram_-con"),
    ).unpivot(
        on=pl.selectors.starts_with("mean_std_nll"),
        index=["step"],
        variable_name="subset",
        value_name="mean_std_nll",
    ).with_columns(
        pl.col("subset").str.replace("mean_std_nll_", "").str.replace("_", " "),
    )

    sns.lineplot(
        _df,
        x="step",
        y="mean_std_nll",
        hue="subset",
        palette="Paired"
    )
    return


@app.cell
def _(dataset_one, pl, sns):
    _df = dataset_one.filter(pl.col("step").ne(0), pl.col("is_lemma_ok")).group_by("step", "file_name", "file_line_idx").agg(
        pl.col("ppl_grammatical", "ppl_ungrammatical").log().mean().name.replace("ppl_", "mean_nll_"),
        pl.col("is_lemma_ok", "is_llm_verdict").first(),
    ).sort("step", "file_name", "file_line_idx").group_by("step").agg(
        pl.col("mean_nll_grammatical").filter("is_llm_verdict").mean().alias("mean_nll_+gram_+con"),
        pl.col("mean_nll_grammatical").filter(pl.col("is_llm_verdict").not_()).mean().alias("mean_nll_+gram_-con"),
        pl.col("mean_nll_ungrammatical").filter("is_llm_verdict").mean().alias("mean_nll_-gram_+con"),
        pl.col("mean_nll_ungrammatical").filter(pl.col("is_llm_verdict").not_()).mean().alias("mean_nll_-gram_-con"),
    ).unpivot(
        on=pl.selectors.starts_with("mean_nll"),
        index=["step"],
        variable_name="subset",
        value_name="mean_nll",
    ).with_columns(
        pl.col("subset").str.replace("mean_nll_", "").str.replace("_", " "),
    )

    sns.lineplot(
        _df,
        x="step",
        y="mean_nll",
        hue="subset",
        palette="Paired"
    )
    return


@app.cell
def _(dataset_one, pl, plt, sns):
    _df = dataset_one.filter(pl.col("is_lemma_ok"), pl.col("step") != 0).group_by("run", "step", pl.col("is_llm_verdict").alias("conceivable")).agg(
        pl.col("correct").mean().alias("accuracy")
    ).with_columns(
        pl.col("run").str.extract(r"^(.*)-\d+$").alias("run_kind")
    )

    _fig, _ax = plt.subplots(figsize=(12, 5))

    _ax = sns.lineplot(
        _df,
        ax=_ax,
        x="step",
        y="accuracy",
        marker="o",
        hue="run_kind",
        style="conceivable",
    )

    _ax.legend_.set_bbox_to_anchor((1.025, 1))
    _ax.set_ylim(0.5, 0.8)
    plt.tight_layout()

    _fig
    return


if __name__ == "__main__":
    app.run()
