# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial

    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl


@app.cell
def _():
    from verb_overgen.train.lib.utils import load_train_config
    from verb_overgen.train.lib.validate import VERB_TARGET_5, VERB_TARGET_17, ci
    from verb_overgen.train.train import TrainingSession

    return TrainingSession, VERB_TARGET_17, ci, load_train_config


@app.cell
def _(TrainingSession, load_train_config):
    _session = TrainingSession(load_train_config())
    tokenizer = _session.setup_tokenizer()._tokenizer
    return (tokenizer,)


@app.cell
def _(pl):
    dataset_result = pl.read_parquet(
        "./data/exp/overgen_results_raw_260226.parquet"
    )
    dataset_result
    return (dataset_result,)


@app.cell
def _(dataset_result, pl):
    dataset_result_agg = (
        dataset_result.group_by("verb", "run_flavor", "kind", "step")
        .agg(
            pl.col("nll").mean().alias("nll_mean"),
            pl.col("nll"),
            # pl.col("nll").map_elements(ci, pl.Array(pl.Float32, 2)).alias("nll_ci"),
        )
        .sort(
            ["verb", "run_flavor", "kind", "step"],
            descending=[False, True, False, False],
        )
    )

    dataset_result_agg
    return (dataset_result_agg,)


@app.cell
def _(dataset_result, pl, tokenizer):
    dataset_token_count = pl.read_parquet(
        "./var/pychildes/ngram/token_count.parquet"
    )

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

    dataset_target_verb_count = pl.DataFrame(
        [
            dict(
                verb=verb,
                count=map_token_count.get(tokenizer.token_to_id(verb), 0),
            )
            for verb in dataset_result.select(pl.col("verb").unique())
            .to_series()
            .to_list()
        ]
    ).sort(by="count", descending=True)
    list_verb_by_count = dataset_target_verb_count["verb"].to_list()
    return (list_verb_by_count,)


@app.cell
def _(VERB_TARGET_17, list_verb_by_count, mo):
    dropdown_verb = mo.ui.dropdown(
        {f"{verb}{'*' if verb in VERB_TARGET_17 else ''}": verb for verb in list_verb_by_count}, value=list_verb_by_count[0]
    )
    checkbox_show_ci = mo.ui.checkbox(
        label="Show 95% Confidence Interval", value=False
    )
    checkbox_show_full = mo.ui.checkbox(
        label="Show Full Training Set Group", value=False
    )

    mo.vstack(
        [
            checkbox_show_ci,
            checkbox_show_full,
        ]
    )
    return checkbox_show_ci, checkbox_show_full, dropdown_verb


@app.cell
def draw_trace(
    alt,
    checkbox_show_ci,
    checkbox_show_full,
    ci,
    dataset_result_agg,
    dropdown_verb,
    mo,
    pl,
):
    _df = dataset_result_agg.filter(pl.col("verb").eq(dropdown_verb.value))

    if not checkbox_show_full.value:
        _df = _df.filter(pl.col("run_flavor") != "full")

    if checkbox_show_ci.value:
        _df = _df.with_columns(
            pl.col("nll").map_elements(ci, pl.Array(pl.Float32, 2)).alias("nll_ci")
        ).with_columns(
            pl.col("nll_ci").arr.get(0).alias("nll_ci_lower"),
            pl.col("nll_ci").arr.get(1).alias("nll_ci_higher"),
        )


    _df = _df.with_columns(
        pl.col("run_flavor").replace(
            {
                "full": "O",
                "remove_preemption": "+(E-P)",
                "remove_entrenchment": "+P",
            }
        ),
        pl.col("kind").str.strip_prefix("sentence_"),
    )

    _color_scale = alt.Scale(
        domain=["O", "+(E-P)", "+P"],
        range=["black", "#FB2C36", "#2B7FFF"],
    )
    _dash_scale = alt.Scale(
        domain=["causative", "transitive"],
        range=[[1, 0], [4, 2]],
    )
    _nll_scale = alt.Scale(domain=[5, 10])

    _base = alt.Chart(_df, title=f"eval/{dropdown_verb.value}").encode(
        x=alt.X("step:Q"),
        y=alt.Y("nll_mean:Q", scale=_nll_scale),
        color=alt.Color(
            "run_flavor:N", scale=_color_scale, legend=alt.Legend(title="Train")
        ),
        strokeDash=alt.StrokeDash(
            "kind:N", scale=_dash_scale, legend=alt.Legend(title="Sentence")
        ),
    )
    _line = _base.mark_line()
    _point = _base.mark_point()

    _chart = _line + _point

    if checkbox_show_ci.value:
        _band = (
            alt.Chart(_df)
            .mark_errorbar(ticks=True)
            .encode(
                x=alt.X("step:Q"),
                y=alt.Y("nll_ci_lower:Q", title="NLL", scale=_nll_scale),
                y2=alt.Y2("nll_ci_higher:Q"),
                color=alt.Color("run_flavor:N", scale=_color_scale),
            )
        )
        _chart += _band

    _chart = _chart.properties(width=800, height=300).interactive()

    mo.vstack(
        [
            dropdown_verb,
            _chart,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
