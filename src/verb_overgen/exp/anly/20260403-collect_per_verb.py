# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")

with app.setup:
    import os

    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt

    os.chdir(os.environ["PWD"])


@app.cell
def _():
    dataset_traj = pl.read_parquet(
        "./data/exp/overgen_trajectory-260402-260403T0658.parquet"
    )
    dataset_traj = dataset_traj.sort("verb", "run_seed", "step")
    # dataset_traj
    return (dataset_traj,)


@app.cell
def _():
    dataset_verb_stats = pl.read_parquet("./var/results/verb_stats.parquet")
    return (dataset_verb_stats,)


@app.cell(hide_code=True)
def _():
    _SCHEMA = """
    Schema([('run_name', String), ('run_subset', String), ('verb', String), ('run_seed', Int64), ('step', Int64), ('epoch', Int64), ('train/verb_feature/this/causative', Int64), ('train/verb_feature/this/causative_entrenched', Int64), ('eval/loss_mean/this/causative', Float64), ('eval/loss_mean/this/transitive', Float64), ('eval/loss_mean/this/context_causative', Float64), ('eval/loss_mean/this/context_transitive', Float64)])
    """
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Eval Loss

    Training set is manipulated per verb.
    """)
    return


@app.cell
def _(dataset_traj):
    _verbs = sorted(dataset_traj["verb"].unique().to_list())
    VERBS_FINAL = _verbs
    dropdown_verb = mo.ui.dropdown(options=_verbs, value=_verbs[0], label="Verb")
    dropdown_verb
    return VERBS_FINAL, dropdown_verb


@app.cell
def _():
    SUBSET_MAP = {
        "remove_preemption_on_verb": "+E",
        "remove_entrenchment_on_verb": "+P",
    }
    _COLOR = alt.Color(
        "run_subset:N",
        title="Subset",
        scale=alt.Scale(
            domain=["+E", "+P"], range=["rgb(209, 77, 65)", "rgb(67, 133, 190)"]
        ),
    )
    _STROKE_DASH = alt.StrokeDash(
        "loss_type:N",
        title="Loss Type",
        scale=alt.Scale(
            domain=["causative", "transitive"], range=[[1, 0], [4, 2]]
        ),
    )


    def prep_loss_df(traj, verb, causative_col, transitive_col):
        return (
            traj.filter(pl.col("verb") == verb)
            .select(
                "step",
                "run_seed",
                pl.col("run_subset").replace_strict(SUBSET_MAP),
                causative_col,
                transitive_col,
            )
            .unpivot(
                on=[causative_col, transitive_col],
                index=["step", "run_subset", "run_seed"],
                variable_name="loss_type",
                value_name="loss",
            )
            .with_columns(
                pl.col("loss_type")
                .str.split("/")
                .list.last()
                .str.replace("context_", "", literal=True)
            )
            .with_columns(
                pl.col("loss")
                .rolling_mean(window_size=5)
                .over("run_subset", "run_seed", "loss_type")
            )
        )


    def prep_diff_df(traj, verb, causative_col, transitive_col):
        return (
            traj.filter(pl.col("verb") == verb)
            .select(
                "step",
                "run_seed",
                pl.col("run_subset").replace_strict(SUBSET_MAP),
                causative_col,
                transitive_col,
            )
            .with_columns(
                (pl.col(transitive_col) - pl.col(causative_col)).alias("loss_diff")
            )
            .select("step", "run_seed", "run_subset", "loss_diff")
            .with_columns(
                pl.col("loss_diff")
                .rolling_mean(window_size=5)
                .over("run_subset", "run_seed")
            )
        )


    def make_loss_chart(df, title):
        _base = alt.Chart(df).encode(x=alt.X("step:Q", title="Step"), color=_COLOR)
        _band = _base.mark_errorband(extent="stdev", opacity=0.35).encode(
            y=alt.Y("loss:Q", title="Loss", scale=alt.Scale(domain=[4, 9])),
            detail=alt.Detail("loss_type:N"),
        )
        _line = _base.mark_line().encode(
            y=alt.Y("mean(loss):Q", title="Loss", scale=alt.Scale(domain=[4, 9])),
            strokeDash=_STROKE_DASH,
            tooltip=[
                "step:Q",
                "run_subset:N",
                "loss_type:N",
                alt.Tooltip("mean(loss):Q", format=".4f"),
                alt.Tooltip("stdev(loss):Q", format=".4f"),
            ],
        )
        return (_band + _line).properties(title=title).interactive()


    def make_diff_chart(df, title):
        _base = alt.Chart(df).encode(x=alt.X("step:Q", title="Step"), color=_COLOR)
        _band = _base.mark_errorband(extent="stdev", opacity=0.35).encode(
            y=alt.Y("loss_diff:Q", title="Transitive − Causative"),
        )
        _line = _base.mark_line().encode(
            y=alt.Y("mean(loss_diff):Q", title="Transitive − Causative"),
            tooltip=[
                "step:Q",
                "run_subset:N",
                alt.Tooltip("mean(loss_diff):Q", format=".4f"),
                alt.Tooltip("stdev(loss_diff):Q", format=".4f"),
            ],
        )
        return (_band + _line).properties(title=title).interactive()

    return (
        SUBSET_MAP,
        make_diff_chart,
        make_loss_chart,
        prep_diff_df,
        prep_loss_df,
    )


@app.cell
def _(dataset_traj, dropdown_verb, make_loss_chart, prep_loss_df):
    make_loss_chart(
        prep_loss_df(
            dataset_traj,
            dropdown_verb.value,
            "eval/loss_mean/this/causative",
            "eval/loss_mean/this/transitive",
        ),
        f"{dropdown_verb.value}",
    )
    return


@app.cell
def _(dataset_traj, dropdown_verb, make_diff_chart, prep_diff_df):
    make_diff_chart(
        prep_diff_df(
            dataset_traj,
            dropdown_verb.value,
            "eval/loss_mean/this/causative",
            "eval/loss_mean/this/transitive",
        ),
        f"Transitive − Causative | {dropdown_verb.value}",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Eval Loss with Context
    """)
    return


@app.cell
def _(dataset_traj, dropdown_verb, make_loss_chart, prep_loss_df):
    make_loss_chart(
        prep_loss_df(
            dataset_traj,
            dropdown_verb.value,
            "eval/loss_mean/this/context_causative",
            "eval/loss_mean/this/context_transitive",
        ),
        f"Context + {dropdown_verb.value}",
    )
    return


@app.cell
def _(dataset_traj, dropdown_verb, make_diff_chart, prep_diff_df):
    make_diff_chart(
        prep_diff_df(
            dataset_traj,
            dropdown_verb.value,
            "eval/loss_mean/this/context_causative",
            "eval/loss_mean/this/context_transitive",
        ),
        f"Context, Transitive − Causative | {dropdown_verb.value}",
    )
    return


@app.cell
def _(dataset_verb_stats):
    dataset_verb_stats
    return


@app.cell
def _(VERBS_FINAL, dataset_verb_stats):
    _chart = (
        alt.Chart(
            dataset_verb_stats.filter(
                pl.col("interest_level") < 4, pl.col("verb").is_in(VERBS_FINAL)
            )
        )
        .mark_circle()
        .encode(
            x=alt.X(
                "count_verb_like:Q",
                title="Verb Count",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "count_causative:Q",
                title="Causative Count",
                scale=alt.Scale(type="symlog", constant=1),
                axis=alt.Axis(grid=False),
            ),
            color=alt.Color("interest_level:N", title="Interest level"),
            tooltip=[
                "verb:N",
                "count_verb_like:Q",
                "count_causative:Q",
                "interest_level:N",
            ],
        )
    )
    _chart
    return


@app.cell
def _(SUBSET_MAP, dataset_traj, dataset_verb_stats):
    dataset_final_loss = (
        dataset_traj.group_by(
            pl.col("verb"),
            pl.col("run_subset").replace_strict(SUBSET_MAP),
            pl.col("run_seed"),
        )
        .agg(
            pl.selectors.starts_with("eval")
            .sort_by("step")
            .last()
            .name.replace("eval/loss_mean/this/", "")
        )
        .unpivot(
            index=["verb", "run_subset", "run_seed"],
            variable_name="loss_type",
            value_name="loss",
        )
    )
    dataset_final_loss_corr = dataset_final_loss.join(
        dataset_verb_stats, on="verb"
    )
    dataset_final_loss_corr
    return (dataset_final_loss_corr,)


@app.cell
def _(dataset_final_loss_corr):
    _chart = (
        alt.Chart(dataset_final_loss_corr.with_columns(
            pl.col("count_causative").truediv(pl.col("count_verb_like")).alias("causative_ratio")
        ))
        .mark_circle()
        .encode(
            x=alt.X("count_verb_like", scale=alt.Scale(type="symlog", constant=1)),
            y="loss",
            color="loss_type",
            tooltip=[
                "verb",
                "count_verb_like",
                "count_non_verb_like",
                "count_causative",
            ],
        )
    )

    _chart
    return


@app.cell
def _(dataset_final_loss_corr):
    _fig, _ax = plt.subplots()
    sns.scatterplot(
        dataset_final_loss_corr.with_columns(
            pl.col("count_causative").truediv(pl.col("count_verb_like")).alias("causative_ratio")
        ),
        x="count_causative",
        y="loss",
        hue="loss_type",
        ax=_ax,
    )
    _ax.set_xscale("log")

    _fig
    return


@app.cell
def _():
    # dataset_final_loss_corr.write_parquet("./var/results/20260403.indep_verb_loss_stats.parquet")
    return


@app.cell
def _(dataset_traj):
    for _run_name in dataset_traj.select(pl.col("run_name").unique()):
        break
    return


if __name__ == "__main__":
    app.run()
