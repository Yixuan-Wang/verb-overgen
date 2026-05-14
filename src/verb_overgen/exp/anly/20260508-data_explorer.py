# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.23.4"
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
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import scipy.stats as stats

    from verb_overgen.util.r_arrow_bridge import RScript
    from verb_overgen.common.features import VERB_TARGET_10

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 14


@app.cell
def setup_dataset():
    dataset_src = pl.scan_parquet("var/results/v2/testset/*.parquet")
    return (dataset_src,)


@app.cell
def _(dataset_src):
    dataset_src.collect_schema()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    Schema([('run_name', String), ('run_seed', Int32), ('run_manipulation', String), ('run_scope', String), ('run_target', String), ('run_target_verb', String), ('ckpt_step', Int32), ('ckpt_epoch', Int32), ('eval_verb', String), ('eval_index_sentence', Int64), ('eval_index_verb_context', UInt32), ('eval_has_context', Boolean), ('eval_construction', String), ('result_token_prob', List(Float32)), ('result_nll', Float32), ('result_slor', Float64)])
    ```
    """)
    return


@app.cell
def display_dataset_info(dataset_src):
    dataset_src_schema = dataset_src.collect_schema()

    assert "run_name" in dataset_src_schema
    assert "run_seed" in dataset_src_schema
    assert "run_manipulation" in dataset_src_schema
    assert "run_scope" in dataset_src_schema
    assert "run_target" in dataset_src_schema
    assert "run_target_verb" in dataset_src_schema
    assert "ckpt_step" in dataset_src_schema
    assert "eval_verb" in dataset_src_schema
    assert "eval_construction" in dataset_src_schema
    assert "eval_has_context" in dataset_src_schema

    summary_run_name: int = dataset_src.select(pl.col("run_name").n_unique()).collect().item()
    summary_run_seed: list[int] = dataset_src.select(pl.col("run_seed").unique()).collect().to_series().sort().to_list()
    summary_run_manipulation: dict[str, int] = dataset_src.group_by("run_manipulation").agg(pl.col("run_name").n_unique()).collect().sort("run_manipulation").pipe(lambda df: dict(zip(df["run_manipulation"], df["run_name"])))
    summary_run_scope: dict[str, int] = dataset_src.group_by("run_scope").agg(pl.col("run_name").n_unique()).collect().sort("run_scope").pipe(lambda df: dict(zip(df["run_scope"], df["run_name"])))
    summary_run_target: dict[str, int] = dataset_src.group_by("run_target").agg(pl.col("run_name").n_unique()).collect().sort("run_target").pipe(lambda df: dict(zip(df["run_target"], df["run_name"])))
    summary_run_target_verb: list[str] = dataset_src.select(pl.col("run_target_verb").unique()).collect().to_series().to_list()
    summary_ckpt_step: list[int] = dataset_src.select(pl.col("ckpt_step").unique()).collect().to_series().sort().to_list()
    summary_eval_verb: int = dataset_src.select(pl.col("eval_verb").n_unique()).collect().item()
    summary_eval_construction: list[str] = dataset_src.select(pl.col("eval_construction").unique()).collect().to_series().sort().to_list()
    summary_eval_has_context: list[bool] = dataset_src.select(pl.col("eval_has_context").unique()).collect().to_series().sort().to_list()

    mo.vstack([
        mo.hstack(row, widths=[1, 2]) for row in [
            [mo.md("run_name"), mo.md(f"*{summary_run_name} unique*")],
            [mo.md("run_seed"), summary_run_seed],
            [mo.md("run_manipulation"), summary_run_manipulation],
            [mo.md("run_scope"), summary_run_scope],
            [mo.md("run_target"), summary_run_target],
            [mo.md("run_target_verb"), summary_run_target_verb],
            [mo.md("ckpt_step"), summary_ckpt_step],
            [mo.md("eval_verb"), mo.md(f"*{summary_eval_verb} unique*")],
            [mo.md("eval_construction"), summary_eval_construction],
            [mo.md("eval_has_context"), summary_eval_has_context],
        ]
    ])
    return


@app.cell
def _(dataset_src):
    dataset_src.select(
        pl.struct("run_manipulation", "run_target", "run_scope").unique().alias("run")
    ).collect().select(
        pl.concat_list(pl.col("run").struct.field("*")).list.filter(pl.element() != "").list.join(" :: ").alias("run")
    )
    return


@app.cell
def _(dataset_src):
    dataset_src.select(
        pl.struct(
            "eval_construction", 
            pl.col("eval_has_context").replace_strict({True: "with_context", False: "no_context"}, return_dtype=pl.String).alias("eval_context"),
        ).unique().alias("eval")
    ).collect().select(
        pl.concat_list(pl.col("eval").struct.field("*")).list.filter(pl.element() != "").list.join(" :: ").alias("eval")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | run |
    |---|
    | remove :: p :: per-verb |
    | remove :: e :: per-verb |
    | remove :: p :: all-verb |
    | remove :: p :: except-verb |
    | remove :: e :: all-verb |
    | none |
    | remove :: e :: except-verb |
    | substitute :: p :: per-verb |

    ---

    | eval |
    |---|
    | causative :: no_context |
    | transitive :: no_context |
    | transitive :: with_context |
    | causative :: with_context |
    """)
    return


@app.cell
def _():
    expr_verb_equal = pl.col("eval_verb").is_in(VERB_TARGET_10) & ((pl.col("run_scope").is_in(["all-verb", ""])) | (pl.col("run_target_verb") == pl.col("eval_verb")))
    expr_epoch_final = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max()
    return expr_epoch_final, expr_verb_equal


@app.cell
def setup_condition_options(dataset_src):
    run_options = (
        dataset_src.select(
            pl.struct("run_manipulation", "run_target", "run_scope").unique().alias("run")
        )
        .collect()
        .with_columns(
            pl.concat_list(pl.col("run").struct.field("*"))
            .list.filter(pl.element() != "")
            .list.join(" :: ")
            .alias("label")
        )
        .sort("label")
    )

    eval_options = (
        dataset_src.select(
            pl.struct(
                "eval_construction",
                pl.col("eval_has_context")
                .replace_strict({True: "with_context", False: "no_context"}, return_dtype=pl.String)
                .alias("eval_context"),
            )
            .unique()
            .alias("eval")
        )
        .collect()
        .with_columns(
            pl.concat_list(pl.col("eval").struct.field("*"))
            .list.filter(pl.element() != "")
            .list.join(" :: ")
            .alias("label")
        )
        .sort("label")
    )
    return eval_options, run_options


@app.cell
def condition_picker(eval_options, run_options):
    ui_run_select = mo.ui.multiselect(
        options=run_options["label"].to_list(),
        label="Run conditions",
    )
    ui_eval_select = mo.ui.multiselect(
        options=eval_options["label"].to_list(),
        label="Eval conditions",
    )
    ui_metric_select = mo.ui.radio(
        options=["NLL", "SLOR"],
        value="NLL",
        label="Metric",
        inline=True,
    )
    mo.vstack([ui_run_select, ui_eval_select, ui_metric_select])
    return ui_eval_select, ui_metric_select, ui_run_select


@app.cell
def resolve_metric(ui_metric_select):
    metric_choice = ui_metric_select.value or "NLL"
    metric_col = {"NLL": "result_nll", "SLOR": "result_slor"}[metric_choice]
    metric_label = metric_choice
    return metric_col, metric_label


@app.cell
def infer_analysis(eval_options, run_options, ui_eval_select, ui_run_select):
    mo.stop(
        not ui_run_select.value or not ui_eval_select.value,
        mo.md("*Pick at least one run and one eval condition to infer the analysis.*"),
    )

    selected_runs = run_options.filter(pl.col("label").is_in(ui_run_select.value))
    selected_evals = eval_options.filter(pl.col("label").is_in(ui_eval_select.value))

    def _classify(df, struct_col, dims, all_options):
        full = {
            d: set(all_options.select(pl.col(struct_col).struct.field(d).unique()).to_series().to_list())
            for d in dims
        }
        sel = {
            d: set(df.select(pl.col(struct_col).struct.field(d).unique()).to_series().to_list())
            for d in dims
        }
        filters, comparisons, marginalized = {}, {}, {}
        for _d in dims:
            if len(sel[_d]) == 1:
                (_val,) = sel[_d]
                filters[_d] = _val if _val != "" else "∅"
            elif sel[_d] == full[_d]:
                marginalized[_d] = sorted(sel[_d])
            else:
                comparisons[_d] = sorted(sel[_d])
        return filters, comparisons, marginalized

    run_filters, run_comparisons, run_marginalized = _classify(
        selected_runs, "run", ["run_manipulation", "run_target", "run_scope"], run_options
    )
    eval_filters, eval_comparisons, eval_marginalized = _classify(
        selected_evals, "eval", ["eval_construction", "eval_context"], eval_options
    )

    def _short(d):
        return d.replace("run_", "").replace("eval_", "")

    def _phrase(d, vs):
        return f"**{_short(d)}** ({' vs '.join(v if v else '∅' for v in vs)})"

    bits = []
    if run_comparisons:
        bits.append("Effect of " + ", ".join(_phrase(k, v) for k, v in run_comparisons.items()))
    if eval_comparisons:
        bits.append(("across " if bits else "Variation across ") + ", ".join(_phrase(k, v) for k, v in eval_comparisons.items()))
    if eval_filters:
        bits.append("on " + ", ".join(f"{_short(k)}={v}" for k, v in eval_filters.items()))
    if run_filters:
        bits.append("holding " + ", ".join(f"{_short(k)}={v}" for k, v in run_filters.items()) + " fixed")
    if run_marginalized or eval_marginalized:
        bits.append("marginalizing over " + ", ".join(_short(k) for k in {**run_marginalized, **eval_marginalized}))

    headline = " ".join(bits) if bits else "Single cell selected — no contrast."

    details = []
    for _label, _d in [("Run filter", run_filters), ("Eval filter", eval_filters)]:
        if _d:
            details.append(f"- {_label}: " + ", ".join(f"`{k}`={v}" for k, v in _d.items()))
    for _label, _d in [("Run comparison", run_comparisons), ("Eval comparison", eval_comparisons)]:
        if _d:
            details.append(f"- {_label}: " + ", ".join(f"`{k}` ∈ {{{', '.join(v if v else '∅' for v in vs)}}}" for k, vs in _d.items()))
    for _label, _d in [("Run marginalized", run_marginalized), ("Eval marginalized", eval_marginalized)]:
        if _d:
            details.append(f"- {_label}: " + ", ".join(f"`{k}` (all {len(vs)} values)" for k, vs in _d.items()))
    return (
        details,
        eval_comparisons,
        headline,
        run_comparisons,
        selected_evals,
        selected_runs,
    )


@app.cell
def filter_dataset(
    dataset_src,
    expr_epoch_final,
    expr_verb_equal,
    selected_evals,
    selected_runs,
):
    run_keys = selected_runs.select(
        pl.col("run").struct.field("run_manipulation"),
        pl.col("run").struct.field("run_target"),
        pl.col("run").struct.field("run_scope"),
    ).lazy()
    eval_keys = selected_evals.select(
        pl.col("eval").struct.field("eval_construction"),
        pl.col("eval").struct.field("eval_context"),
    ).lazy()

    dataset_filtered = (
        dataset_src.filter(
            expr_verb_equal,
            expr_epoch_final,
        ).with_columns(
            pl.col("eval_has_context")
            .replace_strict({True: "with_context", False: "no_context"}, return_dtype=pl.String)
            .alias("eval_context"),
        )
        .join(run_keys, on=["run_manipulation", "run_target", "run_scope"], how="inner")
        .join(eval_keys, on=["eval_construction", "eval_context"], how="inner")
    )
    dataset_filtered.select(pl.len()).collect().item()
    return (dataset_filtered,)


@app.cell
def aggregate_per_verb(dataset_filtered, metric_col):
    def _metric_ci(s: pl.Series) -> pl.Series:
        _res = stats.bootstrap((s.to_numpy(),), np.mean, confidence_level=0.95, method="percentile")
        return pl.Series([{"lo": float(_res.confidence_interval.low), "hi": float(_res.confidence_interval.high)}])

    dataset_agg = (
        dataset_filtered.collect()
        .group_by(
            "run_manipulation",
            "run_target",
            "run_scope",
            "eval_construction",
            "eval_context",
            "eval_verb",
        )
        .agg(
            pl.col(metric_col).mean().alias("metric_mean"),
            pl.col(metric_col).len().alias("n"),
            pl.col(metric_col)
            .map_batches(
                _metric_ci,
                return_dtype=pl.Struct({"lo": pl.Float64, "hi": pl.Float64}),
                returns_scalar=True,
            )
            .alias("metric_ci"),
        )
        .unnest("metric_ci")
        .rename({"lo": "metric_ci_lo", "hi": "metric_ci_hi"})
        .sort("eval_verb", "run_manipulation", "run_target", "run_scope", "eval_construction", "eval_context")
    )
    dataset_agg
    return (dataset_agg,)


@app.cell
def plot_per_verb(
    dataset_agg,
    details,
    eval_comparisons,
    headline,
    metric_label,
    run_comparisons,
):
    if run_comparisons:
        hue_dims = list(run_comparisons.keys())
        facet_dims = list(eval_comparisons.keys())
    else:
        hue_dims = list(eval_comparisons.keys())
        facet_dims = []

    df_plot = dataset_agg
    if hue_dims:
        df_plot = df_plot.with_columns(
            pl.concat_str([pl.col(_d).cast(pl.String) for _d in hue_dims], separator=" :: ").alias("_hue")
        )
        hue_label = " :: ".join(_d.replace("run_", "").replace("eval_", "") for _d in hue_dims)
    else:
        df_plot = df_plot.with_columns(pl.lit("all").alias("_hue"))
        hue_label = None

    if facet_dims:
        df_plot = df_plot.with_columns(
            pl.concat_str([pl.col(_d).cast(pl.String) for _d in facet_dims], separator=" :: ").alias("_facet")
        )
        facet_label = " :: ".join(_d.replace("run_", "").replace("eval_", "") for _d in facet_dims)
    else:
        df_plot = df_plot.with_columns(pl.lit("all").alias("_facet"))
        facet_label = None

    df_pd = df_plot.to_pandas()

    eval_verbs = sorted(df_pd["eval_verb"].unique())
    x_pos = {v: i for i, v in enumerate(eval_verbs)}
    hue_values = sorted(df_pd["_hue"].unique())
    facet_values = sorted(df_pd["_facet"].unique())
    n_hues = len(hue_values)
    n_facets = len(facet_values)
    dodge = 0.8 / max(n_hues, 1)

    fig, axes = plt.subplots(
        1,
        n_facets,
        figsize=(max(8, 0.6 * len(eval_verbs)) * n_facets, 5),
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()
    palette = sns.color_palette(n_colors=n_hues)

    for _fi, _fv in enumerate(facet_values):
        _ax = axes[_fi]
        _facet_df = df_pd[df_pd["_facet"] == _fv]
        for _i, _h in enumerate(hue_values):
            _sub = _facet_df[_facet_df["_hue"] == _h].sort_values("eval_verb")
            if _sub.empty:
                continue
            _x = np.array([x_pos[v] + (_i - (n_hues - 1) / 2) * dodge for v in _sub["eval_verb"]])
            _y = _sub["metric_mean"].to_numpy()
            _yerr = np.vstack([_y - _sub["metric_ci_lo"].to_numpy(), _sub["metric_ci_hi"].to_numpy() - _y])
            _ax.errorbar(_x, _y, yerr=_yerr, fmt="o", capsize=3, color=palette[_i], label=_h)
        _ax.set_xticks(range(len(eval_verbs)))
        _ax.set_xticklabels(eval_verbs, rotation=45, ha="right")
        _ax.set_xlabel("eval_verb")
        if facet_label:
            _ax.set_title(f"{facet_label}: {_fv}")
        if _fi == 0:
            _ax.set_ylabel(f"mean {metric_label} (95% CI)")

    if hue_label:
        axes[-1].legend(title=hue_label, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    mo.vstack([
        mo.md(f"### {headline}\n\n" + "\n".join(details)),
        fig,
    ])
    return


if __name__ == "__main__":
    app.run()
