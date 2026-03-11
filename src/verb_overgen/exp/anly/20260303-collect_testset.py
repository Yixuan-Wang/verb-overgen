# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", css_file="./notebook.css")


@app.cell
def import_modules():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    return alt, mo, np, pl


@app.cell
def _():
    import os as _os
    _os.chdir(_os.environ["PWD"])
    return


@app.cell
def import_exp():
    from verb_overgen.train.lib.validate import nll, ci
    from verb_overgen.train.lib.dataloader_overgen import clean_dataset_overgen
    from verb_overgen.train.lib.validate import VERB_TARGET_17

    return VERB_TARGET_17, ci, clean_dataset_overgen, nll


@app.cell
def load_data(clean_dataset_overgen, nll, pl):
    dataset_test = pl.read_parquet("./var/train/20260212/fast-paper-90/test").filter(pl.col("step") != 0)

    dataset_test_input = pl.read_parquet("./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet")
    dataset_test_input = clean_dataset_overgen(dataset_test_input)

    dataset_test = dataset_test.with_columns(pl.col("token_prob").pipe(nll)).rename({"token_prob": "nll"})

    dataset_test = dataset_test.join(
        dataset_test_input.select(pl.all().exclude("verb", "kind")), on="index_sentence", coalesce=True
    )
    return (dataset_test,)


@app.cell
def compute_grouped(ci, dataset_test, np, pl):
    def _ci(x):
        return ci(np.array(x))

    dataset_grouped = (
        dataset_test.group_by("verb", "epoch", "kind")
        .agg(
            pl.col("nll").mean().alias("nll_mean"),
            pl.col("nll").alias("nll_list"),
        )
        .with_columns(
            pl.col("nll_list").map_elements(_ci, return_dtype=pl.Array(pl.Float64, 2)).alias("nll_ci")
        )
        .drop("nll_list")
        .sort("verb", "epoch", "kind")
        .select(
            pl.all().exclude("nll_ci"),
            pl.col("nll_ci").arr.get(0).alias("nll_ci_lower"),
            pl.col("nll_ci").arr.get(1).alias("nll_ci_upper"),
        )
    )
    return (dataset_grouped,)


@app.cell
def _(dataset_grouped):
    dataset_grouped
    return


@app.cell
def create_selector(VERB_TARGET_17, dataset_grouped, mo, pl):
    verb_options = dataset_grouped.filter(pl.col("verb").is_in(VERB_TARGET_17)).select(pl.col("verb").unique()).to_series().to_list()
    verb_sel = mo.ui.dropdown(options=verb_options, value=verb_options[0], label="Verb")
    return (verb_sel,)


@app.cell
def plot_nll(alt, dataset_grouped, mo, pl, verb_sel):
    kind_domain = ["sentence_causative", "sentence_transitive"]
    dash_scale = alt.Scale(
        domain=kind_domain,
        range=[[1, 0], [8, 4]],  # solid, dashed
    )

    df_verb = dataset_grouped.filter(pl.col("verb") == verb_sel.value)

    _base = alt.Chart(df_verb).encode(
        x=alt.X("epoch:Q", scale=alt.Scale(domain=[0, 9]), axis=alt.Axis(tickMinStep=1)),
        strokeDash=alt.StrokeDash("kind:N", scale=dash_scale, sort=kind_domain, title="kind"),
        detail="kind:N",
    )

    _line = _base.mark_line(point=True, color="black").encode(
        y=alt.Y("nll_mean:Q", title="NLL", scale=alt.Scale(domain=[5.0, 10.0])),
    )

    _error_bars = _base.mark_errorbar(ticks=True, color="black").encode(
        y=alt.Y("nll_ci_lower:Q", title="NLL"),
        y2=alt.Y2("nll_ci_upper:Q"),
    )

    _chart = (_line + _error_bars).properties(
        title=alt.TitleParams(f"NLL by epoch - verb: {verb_sel.value}", anchor="start"),
        width=500,
        height=300,
    )

    mo.vstack([verb_sel, _chart])
    return


@app.cell
def _():
    import tokenizers as hftk

    tokenizer = hftk.Tokenizer.from_file("./data/tokenizer/1024-wordlevel/tokenizer.json")
    return (tokenizer,)


@app.cell
def _(pl):
    dataset_token_count = pl.read_parquet("./var/pychildes/ngram/token_count.parquet")

    dataset_token_count = dataset_token_count.select(
        pl.col("input_ids").alias("token"),
        pl.col("count").alias("token_count"),
    )

    map_token_count = dict(
        zip(
            dataset_token_count["token"],
            dataset_token_count["token_count"],
        )
    )

    # dataset_input_token_count = dataset_input.select(
    #     pl.col("index_sentence"),
    #     pl.col("input_ids")
    #     .list.eval(
    #         pl.element().map_elements(map_token_count.get, return_dtype=pl.Float64)
    #     )
    #     .alias("token_count"),
    # )

    # def slor(token_prob: pl.Expr, token_count: pl.Expr):
    #     return (
    #         token_prob.list.eval(pl.element().cast(pl.Float64).log()).list.sum()
    #         .sub(token_count.list.eval(pl.element().cast(pl.Float64).log()).list.sum())
    #         .truediv(token_prob.list.len())
    #         .alias("slor")
    #     )
    return (map_token_count,)


@app.cell
def _(VERB_TARGET_17, map_token_count, pl, tokenizer):
    dataset_target_verb_count = pl.DataFrame([ dict(verb=verb, count=map_token_count.get(tokenizer.token_to_id(verb), 0)) for verb in VERB_TARGET_17 ]).sort(by="count", descending=True)
    list_verb_by_count = dataset_target_verb_count["verb"].to_list()

    dataset_target_verb_count
    return dataset_target_verb_count, list_verb_by_count


@app.cell
def _():
    import itertools

    return


@app.cell
def _(dataset_grouped, mo, pl):
    _ser_epoch = dataset_grouped.select(pl.col("epoch").unique()).to_series()
    slider_epoch = mo.ui.slider(_ser_epoch.min(), _ser_epoch.max(), value=_ser_epoch.min(), step=1, label="Epoch") # type: ignore
    return (slider_epoch,)


@app.cell
def _(
    alt,
    dataset_grouped,
    dataset_target_verb_count,
    list_verb_by_count,
    mo,
    pl,
    slider_epoch,
):
    _df = (
        dataset_grouped
        .filter(pl.col("epoch") == slider_epoch.value, pl.col("verb").is_in(list_verb_by_count))
        .join(dataset_target_verb_count, on="verb", how="left")
        .sort("count")
        .with_columns(pl.col("kind").str.strip_prefix("sentence_"))
    )


    _base = alt.Chart(_df).encode(
        x=alt.X("count:Q", title="count", scale=alt.Scale(type="log")),
        y=alt.Y("nll_mean:Q", title="nll_mean", scale=alt.Scale(domain=[5.0, 10.0])),
        strokeDash=alt.StrokeDash("kind:N", title="kind"),
        detail="kind:N",
        tooltip=["verb:N", "kind:N", "count:Q", "nll_mean:Q"],
    )

    _line = _base.mark_line(color="black")
    _points = _base.mark_point(size=40, filled=True, color="black")

    _labels = alt.Chart(_df.filter(pl.col("kind") == "causative")).mark_text(
        align="center",
        baseline="middle",
        fontSize=10,
        color="red",
        dy=10,
    ).encode(
        x=alt.X("count:Q", scale=alt.Scale(type="log")),
        y="nll_mean:Q",
        text="verb:N",
        tooltip=["verb:N", "kind:N", "count:Q", "nll_mean:Q", "offset:Q"],
    )

    _chart = (_line + _points + _labels).properties(width=800, height=300, title=alt.TitleParams(f"NLL by Verb Count at Epoch {slider_epoch.value}", anchor="start"))
    # _chart.save('chart.png', ppi=300)

    mo.vstack([
        slider_epoch,
        _chart,
    ])
    return


if __name__ == "__main__":
    app.run()
