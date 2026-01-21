# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
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
    return Literal, alt, np, partial, pl, sns, stats


@app.cell
def _():
    from apfel import not_none
    return (not_none,)


@app.cell
def _(pl):
    RUN_FULL = "celestial-mountain-7"
    RUN_NO_CAUSATIVE = "wobbly-thunder-8"

    dataset_input = pl.read_parquet("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet")

    dataset_test_result_7 = pl.read_parquet("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/var/train/20251230/celestial-mountain-7/test").with_columns(pl.lit(RUN_FULL).alias("run"))

    dataset_test_result_8 = pl.read_parquet("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/var/train/20251230/wobbly-thunder-8/test").with_columns(pl.lit(RUN_NO_CAUSATIVE).alias("run"))

    def filter_aligned_steps(df):
        return df.filter(
            pl.col("step").is_in([19576, 39153, 58731])
        ).with_columns(
            (pl.col("step").sub(1) // 20000).alias("epoch")
        )

    dataset_test_result_7 = filter_aligned_steps(dataset_test_result_7)
    dataset_test_result_8 = filter_aligned_steps(dataset_test_result_8)
    return (
        RUN_FULL,
        RUN_NO_CAUSATIVE,
        dataset_input,
        dataset_test_result_7,
        dataset_test_result_8,
    )


@app.cell
def _(dataset_input, dataset_test_result_7, dataset_test_result_8, np, pl):
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

    dataset = dataset_test_result_7.join(dataset_input, on="index_sentence", how="left").vstack(
        dataset_test_result_8.join(dataset_input, on="index_sentence", how="left")
    )

    dataset = dataset.filter(pl.col("kind") != "sentence_context").with_columns(
        pl.col("token_prob").map_batches(batch_ppl, pl.Float32).alias("ppl"),
        pl.col("token_prob").map_batches(batch_logprob, pl.Float32).alias("logprob"),
    )
    return (dataset,)


@app.cell
def _(pl):
    dataset_verb_stat = pl.concat([
        pl.read_csv("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/data/testset/gen/causative_verb_counts_supplementary.csv"),
        pl.read_csv("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/data/testset/gen/causative_verb_counts.csv")
    ]).rename({"lemma": "verb"})
    return (dataset_verb_stat,)


@app.cell
def _(dataset, pl):
    dataset_ppl = (
        dataset.group_by("run", "epoch", "index").agg(
        pl.col("verb").unique().item(),
        pl.col("vet").unique().item(),
        pl.col("ppl").filter(pl.col("kind") == "sentence_causative").item().alias("ppl_causative"),
        pl.col("ppl").filter(pl.col("kind") == "sentence_transitive").item().alias("ppl_transitive"),
    ).with_columns(
        pl.col("ppl_causative").truediv("ppl_transitive").log().alias("llr")
    ).sort(["run", "epoch", "verb", "index"])
    )

    dataset_ppl
    return (dataset_ppl,)


@app.cell
def _(dataset_verb_stat):
    dataset_verb_order = dataset_verb_stat.sort("count_causative", descending=True)
    return


@app.cell
def _():
    # df_ppl_causative = dataset_ppl.group_by("epoch", "verb", "index").agg(
    #     pl.col("ppl_causative").filter(pl.col("run") == RUN_FULL).item().alias("full"),
    #     pl.col("ppl_causative").filter(pl.col("run") == RUN_NO_CAUSATIVE).item().alias("no_causative"),
    # ).with_columns(
    #     pl.col("full").truediv("no_causative").log().alias("llr_preemption"),
    # )

    # df_verb_ppl_causative = df_ppl_causative.group_by("epoch", "verb").agg(
    #     pl.col("llr_preemption").mean().alias("mean_llr"),
    # ).join(
    #     dataset_verb_order.with_columns(pl.col("count_causative").truediv("count_is_verb").alias("ratio_causative")), on="verb"
    # )

    # _ax = sns.lineplot(
    #     df_verb_ppl_causative,
    #     x="count_causative",
    #     y="mean_llr",
    #     hue="epoch",
    #     marker="o"
    # )
    # _ax.set_xscale("log")

    # _ax
    return


@app.cell
def _(RUN_FULL, RUN_NO_CAUSATIVE, dataset_ppl, pl, sns):
    _df = dataset_ppl.filter(pl.col("vet")).group_by("epoch", "verb", "index").agg(
        pl.col("ppl_transitive").filter(pl.col("run") == RUN_FULL).item().alias("ppl_ogen_full"),
        pl.col("ppl_transitive").filter(pl.col("run") == RUN_NO_CAUSATIVE).item().alias("ppl_ogen_no_preemption"),
    ).with_columns(
        pl.col("ppl_ogen_full").truediv("ppl_ogen_no_preemption").log().alias("llr_preemption"),
    ).sort("verb")


    _grid = sns.FacetGrid(
        _df,
        col="verb",
        col_wrap=5,
    ).map_dataframe(
        sns.boxenplot,
        x="epoch",
        y="llr_preemption",
    )

    for _ax in _grid.axes.flatten():
        _ax.axhline(0, color="red", linestyle="--")

    _grid
    return


@app.cell
def _(Literal, stats):
    def ttest(struct, alternative="less", returns: Literal["t_stat", "p_value"] = "p_value"):
        arr_ppl = struct.to_arrow()
        v0, v1 = arr_ppl.field(0).to_numpy(), arr_ppl.field(1).to_numpy()
        t_stat, p_value = stats.ttest_rel(v0, v1, alternative=alternative)
        return p_value if returns == "p_value" else t_stat
    return (ttest,)


@app.cell
def _(RUN_FULL, RUN_NO_CAUSATIVE, dataset_ppl, pl):
    dataset_preemption_ogen = dataset_ppl.group_by("verb", "index", "epoch").agg(
        pl.col("ppl_transitive").filter(pl.col("run") == RUN_FULL).item().log().alias("nll_ogen_full"),
        pl.col("ppl_transitive").filter(pl.col("run") == RUN_NO_CAUSATIVE).item().log().alias("nll_ogen_no_preemption"),
    )

    dataset_preemption_ogen
    return (dataset_preemption_ogen,)


@app.cell
def _(alt, dataset_verb_stat):
    alt.Chart(dataset_verb_stat).mark_bar().encode(
        x=alt.X("count_is_verb:Q", scale=alt.Scale(type="symlog")),
        y=alt.Y("verb:N", sort='-x'),
    )
    return


@app.cell
def _():
    return


@app.cell
def _(dataset_preemption_ogen, dataset_verb_stat, not_none, pl, sns):
    _df = (
        dataset_preemption_ogen.group_by("epoch", "verb")
        .agg((pl.col("nll_ogen_full") > pl.col("nll_ogen_no_preemption")).mean().alias("preemption_effect"))
        .join(dataset_verb_stat.with_columns(pl.col(name="count_is_verb").rank(descending=True).sub(1).floordiv(10).alias("group_count_is_verb")), on="verb")
    )

    _ax = sns.lineplot(
        _df,
        x="epoch",
        y="preemption_effect",
        hue="group_count_is_verb",
        estimator="mean",
        marker="o",
        palette=sns.color_palette(palette="mako", n_colors=6),
    )

    _ax.set_xticks([0, 1, 2], labels=["0", "1", "2"])
    _ax.set_ylim(0, 1)
    not_none(_ax.legend_).set_alignment("left")
    not_none(_ax.legend_).set_bbox_to_anchor((1.01, 1))

    _ax
    return


@app.cell
def _(dataset_preemption_ogen, dataset_verb_stat, partial, pl, sns, ttest):
    _df = dataset_preemption_ogen.group_by("verb", "index", "epoch").agg(
        (pl.col("nll_ogen_full").item() - pl.col("nll_ogen_no_preemption").item()).alias("llr_preemption")
    ).sort("verb")

    _v = dataset_verb_stat.sort("count_is_verb", descending=True)
    _df = _df.join(_v, on="verb").sort("count_is_verb", descending=True)

    _df_significance = dataset_preemption_ogen.group_by("verb", "epoch").agg(
        pl.struct("nll_ogen_full", "nll_ogen_no_preemption").map_batches(partial(ttest, alternative="two-sided"), pl.Float64, returns_scalar=True).lt(0.05).alias("significant_effect")
    )

    _df = _df.join(_df_significance, on=["verb", "epoch"])

    _counts = {
        row["verb"]: (row["count_causative"], row["count_is_verb"])
        for row in dataset_verb_stat.to_dicts()
    }

    _grid = sns.FacetGrid(
        _df,
        col="verb",
        col_wrap=8,
        hue="significant_effect",
        palette={True: "blue", False: "gray"},
    ).map_dataframe(
        # sns.violinplot,
        sns.lineplot,
        x="epoch",
        y="llr_preemption",
    )

    for verb, ax in zip(_grid.col_names, _grid.axes.flatten()):
        if ax is None:
            continue
        c_caus, c_verb = _counts.get(verb, (0, 0))
        ax.set_title(f"{verb}\nv_c={c_caus} | v={c_verb}")

    for _ax in _grid.axes.flatten():
        _ax.axhline(0.0, color="black", linestyle=":")

    _grid.figure.tight_layout()

    # plt.savefig("20260120-all-causative-removal-preemption-eff.png", dpi=300)
    _grid
    return


@app.cell
def _():
    # expr_nll_full = pl.col("ppl_transitive").sort_by("index").filter(pl.col("run") == RUN_FULL).log().alias("nll_transitive_full")
    # expr_nll_no_causative = pl.col("ppl_transitive").sort_by("index").filter(pl.col("run") == RUN_NO_CAUSATIVE).log().alias("nll_transitive_no_causative")


    # (_df := dataset_ppl.group_by("verb", "epoch").agg(
    #     pl.struct(expr_nll_full, expr_nll_no_causative).map_batches(partial(ttest, returns="t_stat"), return_dtype=pl.Float64, returns_scalar=True).alias("tstat_premption_effect")
    # ).sort("verb", "epoch"))

    # _grid = sns.FacetGrid(
    #     _df,
    #     col="verb",
    #     col_wrap=8,
    # ).map_dataframe(
    #     sns.lineplot,
    #     x="epoch",
    #     y="tstat_premption_effect",
    #     marker="o"
    # )

    # _lower_bound = stats.t.ppf(0.25, df=74)

    # for _ax in _grid.axes.flatten():
    #     _ax.axhline(_lower_bound, color="black", linestyle=":")
    #     _ax.axhline(-_lower_bound, color="black", linestyle=":")

    #     _ax.set_xticks([0, 1, 2], labels=["0", "1", "2"])


    # _grid
    return


if __name__ == "__main__":
    app.run()
