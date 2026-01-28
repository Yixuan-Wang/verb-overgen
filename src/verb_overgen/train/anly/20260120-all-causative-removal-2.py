# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial
    import os
    from typing import Literal

    import apfel
    import marimo as mo
    import polars as pl
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import altair as alt
    import statsmodels.api as sm


    alt.data_transformers.enable("vegafusion")
    # os.chdir(os.environ["PWD"])
    os.chdir("/home/tomyxw/Projects/2025/verb-overgen/verb-overgen")
    return Literal, mo, partial, pl, sns, stats


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
    dataset_verb_stat = pl.concat([
        pl.read_csv("/home/tomyxw/Projects/2025/verb-overgen/verb-overgen/data/testset/gen/causative_verb_counts_supplementary.csv"),
        pl.read_csv("/home/tomyxw/Projects/2025/verb-overgen/verb-overgen/data/testset/gen/causative_verb_counts.csv")
    ]).rename({"lemma": "verb"})
    return (dataset_verb_stat,)


@app.cell
def _(pl, ppl):
    dataset_input = pl.read_parquet("./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet")
    dataset_run = pl.read_parquet("./var/train/20251230/aggregate/testset_20260123.parquet")

    dataset_run = dataset_run.join(dataset_input.filter(pl.col(name="kind") != "sentence_context").select("index_sentence"), on="index_sentence", how="semi")

    dataset_run = dataset_run.select(
        pl.all().exclude("token_prob"),
        pl.col("token_prob").pipe(ppl).alias("ppl"),
        pl.concat_str(
            pl.col("run_kind"),
            pl.lit("-"),
            pl.col("run_seed").cast(pl.String),
        ).alias("run")
    )
    return dataset_input, dataset_run


@app.cell
def _(dataset_input, dataset_run, pl):
    dataset = dataset_run.join(
        dataset_input.select(
            pl.col("index_sentence"),
            pl.col("index").alias("index_pair"),
            pl.col("verb", "vet"),
            pl.col("kind").alias("sentence_kind"),
            pl.col("sentence"),
        ),
        on="index_sentence",
        how="left",
        coalesce=True,
    )
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset, pl):
    dataset_transitive = dataset.filter(pl.col("sentence_kind") == "sentence_transitive")
    return (dataset_transitive,)


@app.cell
def _(dataset, pl, sns):
    dataset_loss = dataset.group_by("run_kind", "run_seed", "step", "sentence_kind").agg(
        pl.col("ppl").log().mean().alias("mean_nll"),
    ).with_columns(
        pl.col("sentence_kind").str.replace_all("sentence_", "")
    )

    _grid = sns.FacetGrid(
        dataset_loss,
        row="sentence_kind",
        col="run_kind",
        aspect=2,
    ).map_dataframe(
        sns.lineplot,
        x="step",
        y="mean_nll",
        hue="run_seed",
        style="sentence_kind",
        palette="tab10",
        marker="o",
    )

    _grid.set_titles("test({row_name}) | train({col_name})")
    return


@app.cell
def _(dataset_transitive, pl):
    dataset_preemption_ogen = dataset_transitive.filter(pl.col("run_seed") != 42).group_by("run_seed", "verb", "index_pair", "step").agg(
        pl.col("ppl").filter(pl.col("run_kind") == "full").item().log().alias("nll_ogen_full"),
        pl.col("ppl").filter(pl.col("run_kind") == "no-causative").item().log().alias("nll_ogen_no_causative"),
    )

    dataset_preemption_ogen
    return (dataset_preemption_ogen,)


@app.cell
def _(Literal, stats):
    def ttest(struct, alternative="less", returns: Literal["t_stat", "p_value"] = "p_value"):
        arr_ppl = struct.to_arrow()
        v0, v1 = arr_ppl.field(0).to_numpy(), arr_ppl.field(1).to_numpy()
        t_stat, p_value = stats.ttest_rel(v0, v1, alternative=alternative)
        return p_value if returns == "p_value" else t_stat
    return (ttest,)


@app.cell
def _(dataset_preemption_ogen):
    dataset_preemption_ogen
    return


@app.cell
def _(dataset_verb_stat):
    verb_counts = {
        row["verb"]: (row["count_causative"], row["count_is_verb"])
        for row in dataset_verb_stat.to_dicts()
    }
    return (verb_counts,)


@app.cell
def _(mo):
    checkbox_show_epoch_only = mo.ui.checkbox(False, label="Show epoch only")
    return (checkbox_show_epoch_only,)


@app.cell
def _():
    verb_allow_list = "appear, chortle, chuckle, frown, giggle, glow, grin, groan, laugh, rise, sing, sleep, smile, vanish, sneeze".split(", ")
    return (verb_allow_list,)


@app.cell
def _(dataset_preemption_ogen, pl):
    dataset_preemption_ogen_epoch = dataset_preemption_ogen.filter(pl.col("step").is_in([19576, 39153, 58730])).with_columns(
        pl.when(pl.col("step") == 19576).then(1)
        .when(pl.col("step") == 39153).then(2)
        .when(pl.col("step") == 58730).then(3)
        .otherwise(pl.lit(None)).alias("epoch")
    )
    return (dataset_preemption_ogen_epoch,)


@app.cell
def _(
    checkbox_show_epoch_only,
    dataset_preemption_ogen,
    dataset_verb_stat,
    mo,
    partial,
    pl,
    sns,
    ttest,
    verb_allow_list,
    verb_counts,
):
    if checkbox_show_epoch_only.value:
        _df = dataset_preemption_ogen.filter(pl.col("step").is_in([19576, 39153, 58730]))
    else:
        _df = dataset_preemption_ogen

    _df = _df.filter(pl.col("verb").is_in(verb_allow_list))

    # _df = _df.filter(pl.col("run_seed") == 37)

    _df = _df.group_by("run_seed","step", "verb", "index_pair").agg(
        (pl.col("nll_ogen_full").item() - pl.col("nll_ogen_no_causative").item()).alias("preemption")
    )

    #.group_by("run_seed", "step", "verb").agg(
    #    pl.col("preemption").mean().alias("mean_preemption")
    #)

    _df_significance = dataset_preemption_ogen.group_by("run_seed","step", "verb").agg(
        pl.struct("nll_ogen_full", "nll_ogen_no_causative").map_batches(partial(ttest, alternative="greater"), pl.Float64, returns_scalar=True).lt(0.05).alias("significant_effect")
    )

    _df = _df.join(_df_significance, on=["run_seed", "verb", "step"])

    _v = dataset_verb_stat.sort(pl.col("count_causative"), descending=True).select("verb").with_row_index("rank")

    _df = _df.join(_v, on="verb").sort("rank")

    if checkbox_show_epoch_only.value:
        _df = _df.rename({"step": "epoch"})

    _grid = sns.FacetGrid(
        _df,
        col="verb",
        col_wrap=5,
        hue="run_seed",
        palette="mako_r",
    ).map_dataframe(
        sns.lineplot,
        x="step" if not checkbox_show_epoch_only.value else "epoch",
        y="preemption",
        # units="run_seed",
        marker="o",
        # estimator=None,
    )

    _grid.set_titles("{col_name}")

    for _verb, _ax in zip(_grid.col_names, _grid.axes.flatten()):
        if _ax is None:
            continue
        _c_caus, _c_verb = verb_counts.get(_verb, (0, 0))
        _ax.set_title(f"{_verb}\nv_c={_c_caus} | v={_c_verb}")

    for _ax in _grid.axes.flatten():
        _ax.axhline(0.0, color="black", linestyle=":")
        if checkbox_show_epoch_only.value:
            _ax.set_xticks([19576, 39153, 58730], labels=["1", "2", "3"])
        # for _label in _ax.get_xticklabels():
            # _label.set_rotation(45)
            # _label.set_ha("right")

    _grid.figure.tight_layout()

    mo.vstack([
        checkbox_show_epoch_only,
        _grid,
        _df,
    ])
    return


@app.cell
def _(dataset_preemption_ogen_epoch, pl, verb_allow_list):
    df_preemption = dataset_preemption_ogen_epoch.filter(pl.col("verb").is_in(verb_allow_list))
    return (df_preemption,)


@app.cell
def _():
    palette_slides = ["#595959", "#3c78d8"]
    return


@app.cell
def _():
    # _df = df_preemption.filter(
    #     pl.col("verb") == "laugh",
    #     pl.col("run_seed") == 37,
    # ).unpivot(
    #     index=["index_pair", "epoch"],
    #     on=["nll_ogen_full", "nll_ogen_no_causative"],
    #     variable_name="train_set",
    #     value_name="nll",
    # ).with_columns(
    #     pl.col("train_set").str.replace_all("nll_ogen_", "")
    # ).with_columns(
    #     pl.when(pl.col("train_set") == "no_causative").then(pl.lit("no_preemption")).otherwise(pl.col("train_set")).alias("train_set")
    # )

    # _ax =sns.violinplot(
    #     _df,
    #     x="epoch",
    #     y="nll",
    #     hue="train_set",
    #     palette=palette_slides,
    #     # density_norm="count",
    # )

    # not_none(_ax.legend_).set_bbox_to_anchor((1.025, 1))

    # _ax
    return


@app.cell
def _(df_preemption, pl, sns):
    _df = df_preemption.filter(
        pl.col("verb") == "laugh",
        pl.col("run_seed") == 23,
    ).with_columns(
        # pl.col("nll_ogen_full").neg().exp().alias("p_full"),
        # pl.col("nll_ogen_no_causative").neg().exp().alias("p_no_preemption"),
        pl.col("nll_ogen_full").alias("nll_full"),
        pl.col("nll_ogen_no_causative").alias("nll_no_preemption"),
    )

    _ax = sns.lmplot(
        _df,  # type: ignore
        # x="p_no_preemption",
        x="nll_no_preemption",
        # y="p_full",
        y="nll_full",
        hue="epoch",
    )

    _ax
    return


@app.cell
def _(df_preemption, pl):
    _df = df_preemption.unpivot(
        index=["index_pair", "epoch", "verb", "run_seed"],
        on=["nll_ogen_full", "nll_ogen_no_causative"],
        variable_name="train_set",
        value_name="nll",
    ).with_columns(
        pl.col("train_set").str.replace_all("nll_ogen_", "")
    ).with_columns(
        pl.when(pl.col("train_set") == "no_causative").then(pl.lit("no_preemption")).otherwise(pl.col("train_set")).alias("train_set")
    )

    _df.write_parquet("./var/result/df_preemption_nll.parquet")
    return


@app.cell
def _(dataset, dataset_verb_stat, mo, partial, pl, sns, ttest, verb_counts):
    dataset_causative = dataset.filter(pl.col("sentence_kind") == "sentence_causative")

    dataset_nll_causative = dataset_causative.filter(pl.col("run_seed") != 42, pl.col("step").is_in([19576, 39153, 58730])).group_by("run_seed", "verb", "index_pair", "step").agg(
        pl.col("ppl").filter(pl.col("run_kind") == "full").item().log().alias("nll_ogen_full"),
        pl.col("ppl").filter(pl.col("run_kind") == "no-causative").item().log().alias("nll_ogen_no_causative"),
    )

    _df = dataset_nll_causative.group_by("run_seed","step", "verb", "index_pair").agg(
        (pl.col("nll_ogen_full").item() - pl.col("nll_ogen_no_causative").item()).alias("preemption")
    ).group_by("run_seed", "step", "verb").agg(
        pl.col("preemption").mean().alias("mean_preemption")
    )

    _df_significance = dataset_nll_causative.group_by("run_seed","step", "verb").agg(
        pl.struct("nll_ogen_full", "nll_ogen_no_causative").map_batches(partial(ttest, alternative="greater"), pl.Float64, returns_scalar=True).lt(0.05).alias("significant_effect")
    )

    _df = _df.join(_df_significance, on=["run_seed", "verb", "step"])

    _v = dataset_verb_stat.sort(pl.col("count_causative"), descending=True).select("verb").with_row_index("rank")

    _df = _df.join(_v, on="verb").sort("rank")

    _grid = sns.FacetGrid(
        _df,
        col="verb",
        col_wrap=8,
    ).map_dataframe(
        sns.lineplot,
        x="step",
        y="mean_preemption",
        # units="run_seed",
        marker="o",
        color=sns.color_palette("tab10")[1],
        # estimator=None,
    )


    for _verb, _ax in zip(_grid.col_names, _grid.axes.flatten()):
        if _ax is None:
            continue
        _c_caus, _c_verb = verb_counts.get(_verb, (0, 0))
        _ax.set_title(f"{_verb}\nv_c={_c_caus} | v={_c_verb}")


    for _ax in _grid.axes.flatten():
        _ax.axhline(0.0, color="black", linestyle=":")
        for _label in _ax.get_xticklabels():
            _label.set_rotation(45)
            _label.set_ha("right")

    _grid.figure.tight_layout()

    mo.vstack([
        _grid,
        _df,
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    # _df = dataset_preemption_ogen.group_by("run_seed","step", "verb", "index_pair").agg(
    #     (pl.col("nll_ogen_full").item() - pl.col("nll_ogen_no_causative").item()).alias("llr_preemption")
    # ).sort("verb")

    # _v = dataset_verb_stat.sort("count_is_verb", descending=True)
    # _df = _df.join(_v, on="verb").sort("count_is_verb", descending=True)

    # _df_significance = dataset_preemption_ogen.group_by("run_seed", "verb", "step").agg(
    #     pl.struct("nll_ogen_full", "nll_ogen_no_causative").map_batches(partial(ttest, alternative="two-sided"), pl.Float64, returns_scalar=True).lt(0.05).alias("significant_effect")
    # )

    # _df = _df.join(_df_significance, on=["run_seed", "verb", "step"])
    # _counts = {
    #     row["verb"]: (row["count_causative"], row["count_is_verb"])
    #     for row in dataset_verb_stat.to_dicts()
    # }

    # _grid = sns.FacetGrid(
    #     _df,
    #     col="verb",
    #     col_wrap=8,
    #     # hue="significant_effect",
    #     # palette={True: "blue", False: "gray"},
    # ).map_dataframe(
    #     # sns.violinplot,
    #     sns.lineplot,
    #     x="step",
    #     y="llr_preemption",
    # )

    # for verb, ax in zip(_grid.col_names, _grid.axes.flatten()):
    #     if ax is None:
    #         continue
    #     c_caus, c_verb = _counts.get(verb, (0, 0))
    #     ax.set_title(f"{verb}\nv_c={c_caus} | v={c_verb}")

    # for (_, _data), _ax in zip(_grid.facet_data(), _grid.axes.flatten()):
    #     sns.scatterplot(
    #         data=_data,
    #         x="step",
    #         y="llr_preemption",
    #         hue="significant_effect",
    #         palette={True: "blue", False: "gray"},
    #         ax=_ax,
    #         legend=False,
    #         s=20,
    #     )

    # for _ax in _grid.axes.flatten():
    #     _ax.axhline(0.0, color="black", linestyle=":")

    # _grid.figure.tight_layout()

    # # plt.savefig("20260120-all-causative-removal-preemption-eff.png", dpi=300)
    # mo.vstack([
    #     _grid,
    #     _df,
    # ])
    return


if __name__ == "__main__":
    app.run()
