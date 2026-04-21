# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.23.2"
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

    from typing import Literal

    import marimo as mo
    import polars as pl
    from adjustText import adjust_text
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import altair as alt
    import scipy.stats as stats

    from verb_overgen.util.r_arrow_bridge import RScript

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 14


@app.cell
def _():
    MAP_COLUMN_CATEGORY: dict[str, Literal["RUN", "POINT", "INPUT", "METRIC"]] = {
        "run_name": "RUN",
        "run_seed": "RUN",
        "run_subset": "RUN",
        "run_subset_verb_level": "RUN",
        "epoch": "RUN",
        "verb": "RUN",
        "index_sentence": "POINT",
        "index_verb_context": "POINT",
        "sentence_construction": "POINT",
        "sentence_has_context": "POINT",
        "sentence": "INPUT",
        "lemma": "INPUT",
        "input_ids": "INPUT",
        "token_prob": "METRIC",
        "nll": "METRIC",
    }
    return (MAP_COLUMN_CATEGORY,)


@app.cell
def _():
    dataset_stats = pl.read_parquet("./var/results/verb_stats.parquet")
    dataset_stats.filter(
        pl.col("verb").is_in([
            "laugh", "cry", "sing", "sleep", "smile", "go", "look", "sneeze", "blink", "rise", "glow",
        ])
    ).select(
        "verb", "count_causative", "count_verb_like"
    )
    return (dataset_stats,)


@app.cell
def _(
    MAP_COLUMN_CATEGORY: dict[str, Literal["RUN", "POINT", "INPUT", "METRIC"]],
):
    dataset = pl.read_parquet(
        "./data/exp/shared/overgen_results-260417T0354.parquet"
    )

    dataset = dataset.with_columns(
        pl.col("run_subset")
        .str.split("-")
        .list.get(0)
        .replace(
            {
                "remove_preemption": "-P",
                "remove_entrenchment": "+P",
            }
        )
        .alias("run_subset"),
        pl.col("run_subset")
        .str.split("-")
        .list.get(1)
        .alias("run_subset_verb_level"),
        pl.col("sentence_construction")
        .str.strip_prefix("context_")
        .alias("sentence_construction"),
        pl.col("sentence_construction")
        .str.starts_with("context_")
        .alias("sentence_has_context"),
    )

    assert set(dataset.columns) == set(MAP_COLUMN_CATEGORY.keys()), (
        f"Expected columns: {set(MAP_COLUMN_CATEGORY.keys())}, but got: {set(dataset.columns)}"
    )

    dataset = dataset.select(MAP_COLUMN_CATEGORY.keys())
    return (dataset,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Peeking Dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```plaintext
    shape: (148_080, 15)
    ┌────────────────────┬──────────┬────────────┬───────────────────────┬───────┬───────┬────────────────┬────────────────────┬───────────────────────┬──────────────────────┬─────────────────────────────────┬────────────────────────────┬────────────────┬─────────────────────────────────┬──────────┐
    │ run_name           ┆ run_seed ┆ run_subset ┆ run_subset_verb_level ┆ epoch ┆ verb  ┆ index_sentence ┆ index_verb_context ┆ sentence_construction ┆ sentence_has_context ┆ sentence                        ┆ lemma                      ┆ input_ids      ┆ token_prob                      ┆ nll      │
    │ ---                ┆ ---      ┆ ---        ┆ ---                   ┆ ---   ┆ ---   ┆ ---            ┆ ---                ┆ ---                   ┆ ---                  ┆ ---                             ┆ ---                        ┆ ---            ┆ ---                             ┆ ---      │
    │ str                ┆ i32      ┆ str        ┆ str                   ┆ i32   ┆ str   ┆ i64            ┆ u32                ┆ str                   ┆ bool                 ┆ str                             ┆ list[str]                  ┆ list[i64]      ┆ list[f32]                       ┆ f32      │
    ╞════════════════════╪══════════╪════════════╪═══════════════════════╪═══════╪═══════╪════════════════╪════════════════════╪═══════════════════════╪══════════════════════╪═════════════════════════════════╪════════════════════════════╪════════════════╪═════════════════════════════════╪══════════╡
    │ firm-aardvark-192  ┆ 597      ┆ -P         ┆ per_verb              ┆ 0     ┆ go    ┆ 1191           ┆ 0                  ┆ causative             ┆ false                ┆ Eleanor makes the rocket go.    ┆ ["Eleanor", "make", … "."] ┆ [1, 364, … 4]  ┆ [0.000186, 0.003845, … 0.05843… ┆ 5.605957 │
    │ firm-aardvark-192  ┆ 597      ┆ -P         ┆ per_verb              ┆ 0     ┆ go    ┆ 1192           ┆ 1                  ┆ causative             ┆ false                ┆ Matthew makes the truck go.     ┆ ["Matthew", "make", … "."] ┆ [1, 1057, … 4] ┆ [0.00004, 0.003319, … 0.072605… ┆ 5.268573 │
    │ firm-aardvark-192  ┆ 597      ┆ -P         ┆ per_verb              ┆ 0     ┆ go    ┆ 1193           ┆ 2                  ┆ causative             ┆ false                ┆ Brett makes the shark go.       ┆ ["Brett", "make", … "."]   ┆ [1, 2150, … 4] ┆ [0.000001, 0.001714, … 0.06002… ┆ 6.37233  │
    │ firm-aardvark-192  ┆ 597      ┆ -P         ┆ per_verb              ┆ 0     ┆ go    ┆ 1194           ┆ 3                  ┆ causative             ┆ false                ┆ Caitlin makes the airplane go.  ┆ ["Caitlin", "make", … "."] ┆ [1, 2690, … 4] ┆ [0.000007, 0.003745, … 0.07484… ┆ 6.253979 │
    │ firm-aardvark-192  ┆ 597      ┆ -P         ┆ per_verb              ┆ 0     ┆ go    ┆ 1195           ┆ 4                  ┆ causative             ┆ false                ┆ Amy makes Nina go.              ┆ ["Amy", "make", … "."]     ┆ [1, 487, … 4]  ┆ [0.000247, 0.003735, … 0.08877… ┆ 6.156525 │
    │ …                  ┆ …        ┆ …          ┆ …                     ┆ …     ┆ …     ┆ …              ┆ …                  ┆ …                     ┆ …                    ┆ …                               ┆ …                          ┆ …              ┆ …                               ┆ …        │
    │ laced-elevator-343 ┆ 189      ┆ +P         ┆ all_verb              ┆ 2     ┆ blink ┆ 7010           ┆ 70                 ┆ transitive            ┆ true                 ┆ Gillian blows a soft puff of a… ┆ ["Gillian", "blow", … "."] ┆ [1, 2667, … 4] ┆ [0.000002, 0.000127, … 0.58030… ┆ 8.313287 │
    │ laced-elevator-343 ┆ 189      ┆ +P         ┆ all_verb              ┆ 2     ┆ blink ┆ 7011           ┆ 71                 ┆ transitive            ┆ true                 ┆ Julian shines a bright flashli… ┆ ["Julian", "shine", … "."] ┆ [1, 2469, … 4] ┆ [0.000006, 0.0001, … 0.606582]  ┆ 5.833494 │
    │ laced-elevator-343 ┆ 189      ┆ +P         ┆ all_verb              ┆ 2     ┆ blink ┆ 7012           ┆ 72                 ┆ transitive            ┆ true                 ┆ Danny blows a soft puff of air… ┆ ["Danny", "blow", … "."]   ┆ [1, 1880, … 4] ┆ [0.000021, 0.000113, … 0.3867]  ┆ 7.412767 │
    │ laced-elevator-343 ┆ 189      ┆ +P         ┆ all_verb              ┆ 2     ┆ blink ┆ 7013           ┆ 73                 ┆ transitive            ┆ true                 ┆ Mike blows a gentle puff of ai… ┆ ["Mike", "blow", … "."]    ┆ [1, 1389, … 4] ┆ [0.000021, 0.000095, … 0.74580… ┆ 6.73883  │
    │ laced-elevator-343 ┆ 189      ┆ +P         ┆ all_verb              ┆ 2     ┆ blink ┆ 7014           ┆ 74                 ┆ transitive            ┆ true                 ┆ Thomas shines a bright flashli… ┆ ["Thomas", "shine", … "."] ┆ [1, 128, … 4]  ┆ [0.000744, 0.000014, … 0.54477… ┆ 5.526484 │
    └────────────────────┴──────────┴────────────┴───────────────────────┴───────┴───────┴────────────────┴────────────────────┴───────────────────────┴──────────────────────┴─────────────────────────────────┴────────────────────────────┴────────────────┴─────────────────────────────────┴──────────┘

    ```
    """)
    return


@app.cell
def _(dataset):
    with pl.Config(tbl_width_chars=-1, tbl_cols=-1):
        print(dataset)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Effect of +P/-P
    """)
    return


@app.cell
def _():
    epoch_selector = mo.ui.dropdown(
        options={"Last": "last", "0": 0, "1": 1, "2": 2},
        value="Last",
        label="Epoch",
    )
    context_toggle = mo.ui.switch(value=False, label="Sentence has context")
    mo.hstack([epoch_selector, context_toggle])
    return context_toggle, epoch_selector


@app.cell
def _(context_toggle, epoch_selector):
    epoch_filter = (
        pl.col("epoch").eq(pl.col("epoch").max())
        if epoch_selector.value == "last"
        else pl.col("epoch").eq(epoch_selector.value)
    )
    context_filter = (
        pl.col("sentence_has_context")
        if context_toggle.value
        else pl.col("sentence_has_context").not_()
    )
    context_label = "with context" if context_toggle.value else "without context"
    return context_filter, context_label, epoch_filter


@app.cell
def _(context_filter, dataset, epoch_filter):
    dataset_base_condition = dataset.filter(
        pl.col("run_subset_verb_level").eq("per_verb"),
        epoch_filter,
        context_filter,
        pl.col("sentence_construction").eq("transitive"),
    ).select("run_seed", "verb", "index_verb_context", "run_subset", "nll")
    return (dataset_base_condition,)


@app.cell
def _():
    def _get_has_preemption_and_no_preemption_nll(
        df: pl.DataFrame,
    ) -> tuple[np._typing.NDArray[np.float64], np._typing.NDArray[np.float64]]:
        _arr_has_preemption = df.filter(pl.col("run_subset").eq("+P"))[
            "nll"
        ].to_numpy()
        _arr_no_preemption = df.filter(pl.col("run_subset").eq("-P"))[
            "nll"
        ].to_numpy()
        return _arr_has_preemption, _arr_no_preemption

    return


@app.cell
def _(dataset_base_condition):
    dataset_base_condition.group_by("verb").agg(
        pl.col("index_verb_context").n_unique()
    ).mean()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mean Transitive NLL by Causative Ratio
    """)
    return


@app.cell
def _(context_filter, context_label, dataset, dataset_stats, epoch_filter):
    _df_plot = (
        dataset.filter(
        pl.col("run_subset_verb_level").eq("per_verb"),
        epoch_filter,
        context_filter,
        pl.col("sentence_construction").eq("transitive"),
    ).select("run_seed", "verb", "index_verb_context", "run_subset", "nll").filter(pl.col("verb") != "glow")
        .select("verb", "run_seed", "run_subset", "nll")
        .join(
            dataset_stats.select(
                "verb",
                (pl.col("count_causative") / pl.col("count_verb_like")).alias(
                    "causative_ratio"
                ),
            ),
            on="verb",
        )
    )

    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.pointplot(
        data=_df_plot,
        x="causative_ratio",
        y="nll",
        hue="run_subset",
        palette={"+P": "steelblue", "-P": "tomato"},
        errorbar="se",
        native_scale=True,
        linestyle="none",
        markers="o",
        markersize=5,
        capsize=0.03,
        ax=_ax,
    )
    _df_plot_label = (
        _df_plot.filter(pl.col("run_subset").eq("+P"))
        .group_by("verb")
        .agg(pl.col("causative_ratio").unique().item(), pl.col("nll").mean())
    )
    _labels = [
        _ax.text(_x * 1.025, _y, _label, ha="left", va="center")
        for _x, _y, _label in zip(
            _df_plot_label["causative_ratio"],
            _df_plot_label["nll"],
            _df_plot_label["verb"],
        )
    ]

    _ax.set_xscale("log")
    _ax.set_xlabel("Causative ratio")
    _ax.set_ylabel(f"Mean NLL on Transitive {context_label}")

    adjust_text(_labels)

    # _ax.set_title("")
    _ax.legend(title="Training Data")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Emmeans
    """)
    return


@app.function
def plot_emmeans_contrasts(
    df_results: pl.DataFrame,
    groups: pl.Series | None = None,
    group_col: str = "verb",
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = (-0.25, 0.25),
    ax=None,
):
    """
    Generates a forest plot from tidied emmeans contrasts.

    Expected columns in df_results:
    - group_col: The grouping factor (default 'verb')
    - 'contrast': The specific pairwise comparison
    - 'estimate': The point estimate of the difference
    - 'std.error': The standard error for the estimate
    """

    if groups is None:
        groups = df_results[group_col].unique().sort()

    contrasts = df_results["contrast"].unique()
    ci_multiplier = 1.96

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(groups) * 0.6)))
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10(np.linspace(0, 1, len(contrasts)))

    group_index = {v: i for i, v in enumerate(groups)}
    dodge_width = 0.6 / max(1, len(contrasts))

    for i, (contrast_name, color) in enumerate(zip(contrasts, colors)):
        subset = df_results.filter(pl.col("contrast") == contrast_name)

        y_pos = [
            group_index[v] + (i * dodge_width)
            for v in subset[group_col]
        ]

        x = subset["estimate"].to_list()
        xerr = (subset["std.error"] * ci_multiplier).to_list()

        ax.errorbar(
            x=x,
            y=y_pos,
            xerr=xerr,
            fmt="o",
            color=color,
            label=contrast_name,
            capsize=4,
            markersize=6,
            linewidth=2,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ytick_offset = dodge_width * (len(contrasts) - 1) / 2
    ax.set_yticks([i + ytick_offset for i in range(len(groups))])
    ax.set_yticklabels(groups.to_list())

    ax.set_xlabel(xlabel if xlabel is not None else "Contrast Estimate")
    ax.set_ylabel(ylabel if ylabel is not None else group_col)

    if len(contrasts) > 1:
        ax.legend(title="Contrast", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(axis="x", linestyle=":", alpha=0.6)
    return fig, ax


@app.cell
def _():
    calc_contrast = RScript("""
    library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

    entry <- function(df) {
        # 1. Use "go" as the reference level for the verb factor
        df <- df %>%
            as_tibble() %>%
            mutate(verb = relevel(factor(verb), ref = "go"))

        # 2. Fit a linear mixed-effects model with the specified formula
        model <- lmer(nll ~ run_subset * verb + (1 | run_seed:index_context), data = df)

        # 3. Compute marginal means, extract contrasts, and tidy the output
        model %>%
            emmeans(revpairwise ~ run_subset | verb) %>%
            .$contrasts %>%
            tidy()
    }
    """)
    return (calc_contrast,)


@app.cell
def _(calc_contrast, context_filter, dataset, dataset_stats, epoch_filter):
    ser_verb_ordering = (
        dataset.unique("verb")
        .join(dataset_stats, on="verb")
        .sort(
            pl.col("count_causative") / pl.col("count_verb_like"), descending=True
        )["verb"]
    )

    _fig, _ax = (
        dataset.filter(
            pl.col("run_subset_verb_level").eq("per_verb"),
            epoch_filter,
            context_filter,
            pl.col("sentence_construction").eq("transitive"),
            pl.col("verb").ne("glow"),
        )
        .select(
            "run_seed",
            "verb",
            pl.struct("verb", "index_verb_context").hash().alias("index_context"),
            "run_subset",
            "nll",
        )
        .pipe(calc_contrast.call_table)
        .pipe(plot_emmeans_contrasts, groups=ser_verb_ordering.filter(ser_verb_ordering != "glow"))
    )
    _fig
    return


@app.cell
def _():
    calc_contrast_level = RScript("""
    library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

    entry <- function(df) {
      df <- df %>%
        as_tibble() %>%
        mutate(
          verb = relevel(factor(verb), ref = "go"),
          run_subset_verb_level = relevel(factor(run_subset_verb_level), ref = "all_verb")
        )

      # 2. Fit a linear mixed-effects model with the specified formula
      model <- lmer(nll ~ run_subset_verb_level * verb + (1 | run_seed:index_context), data = df)

      # 3. Compute marginal means, extract contrasts, and tidy the output
      model %>%
        emmeans(pairwise ~ run_subset_verb_level | verb) %>%
        .$contrasts %>%
        tidy()
    }
    """)
    return (calc_contrast_level,)


@app.cell
def _(
    calc_contrast_level,
    context_filter,
    dataset,
    dataset_stats,
    epoch_filter,
):
    ser_verb_ordering_causative = (
        dataset.unique("verb")
        .join(dataset_stats, on="verb")
        .sort(
            pl.col("count_causative"), descending=True
        )["verb"]
    )

    _df_contrast_level = (
        dataset.filter(
            epoch_filter,
            pl.col("run_subset").eq("-P"),
            context_filter,
            pl.col("sentence_construction").eq("transitive"),
            # pl.col("verb").ne("glow"),
        )
        .select(
            "run_seed",
            "run_subset_verb_level",
            "verb",
            pl.struct("verb", "index_verb_context").hash().alias("index_context"),
            "run_subset",
            "nll",
        )
        .pipe(calc_contrast_level.call_table)
    )

    _fig, _ax = _df_contrast_level.pipe(plot_emmeans_contrasts, groups=ser_verb_ordering_causative)


    # mo.vstack([
    #     _df_contrast_level,
    #     _fig,
    # ])
    _fig
    return


@app.cell
def _(calc_contrast, context_filter, dataset, dataset_stats):
    _run_subset_verb_levels = ["per_verb", "all_verb"]
    _epochs = [0, 1, 2]
    _verbs = (
        dataset.unique("verb")
        .filter(pl.col("verb") != "glow")
        .join(dataset_stats, on="verb")
        .sort(
            pl.col("count_causative") / pl.col("count_verb_like"), descending=True
        )["verb"]
    )

    _fig, _axes = plt.subplots(
        nrows=len(_run_subset_verb_levels),
        ncols=len(_epochs),
        figsize=(
            8 * len(_epochs),
            max(4, len(_verbs) * 0.4) * len(_run_subset_verb_levels),
        ),
    )

    for _i, _run_subset_verb_level in enumerate(_run_subset_verb_levels):
        for _j, _epoch in enumerate(_epochs):
            (
                dataset.filter(
                    pl.col("run_subset_verb_level").eq(_run_subset_verb_level),
                    pl.col("epoch").eq(_epoch),
                    context_filter,
                    pl.col("sentence_construction").eq("transitive"),
                    pl.col("verb").ne("glow"),
                )
                .select(
                    "run_seed",
                    "verb",
                    pl.struct("verb", "index_verb_context")
                    .hash()
                    .alias("index_context"),
                    "run_subset",
                    "nll",
                )
                .pipe(calc_contrast.call_table)
                .pipe(plot_emmeans_contrasts, groups=_verbs, ax=_axes[_i, _j])
            )
            _axes[_i, _j].set_title(
                f"{_run_subset_verb_level}, epoch {_epoch}", loc="left"
            )

    plt.tight_layout()
    _axes[0, 2]
    # _fig.savefig("./var/results/emmeans_contrasts.png", dpi=300)
    return


if __name__ == "__main__":
    app.run()
