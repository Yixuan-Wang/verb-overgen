# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    import subprocess

    os.chdir(os.environ["PWD"])
    if "R_HOME" not in os.environ:
        _direnv_env = subprocess.check_output(
            ["direnv", "export", "json"], text=True
        )
        if _direnv_env:
            os.environ.update(json.loads(_direnv_env))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from verb_overgen.common.features import VERB_TARGET_10

    TESTSET_GLOB = "var/results/v2/testset/*.parquet"
    VERB_STATS_PATH = "var/results/verb_stats.parquet"
    N_BOOT = 2000
    BOOTSTRAP_SEED = 20260514

    METRICS = {
        "NLL": "result_nll",
        "SLOR": "result_slor",
    }

    CONDITION_COLORS = {
        "remove::e::per-verb": "#1f77b4",
        "remove::p::per-verb": "#b22222",
        "substitute::p::per-verb": "#2ca02c",
    }

    CONDITION_LABELS = {
        "remove::e::per-verb": "remove::e::per-verb",
        "remove::p::per-verb": "remove::p::per-verb",
        "substitute::p::per-verb": "substitute::p::per-verb",
    }

    PANEL_ORDER = [
        ("transitive", "NLL"),
        ("transitive", "SLOR"),
        ("causative", "NLL"),
        ("causative", "SLOR"),
    ]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Cross-construction transfer: descriptive per-verb view

    This notebook starts with a descriptive plot of final-epoch
    `with_context` NLL/SLOR by verb. The four plotted condition/eval cells
    are:

    - `transitive :: with_context`: `remove::p::per-verb`,
      `remove::e::per-verb`
    - `causative :: with_context`: `remove::p::per-verb`,
      `substitute::p::per-verb`

    Error bars are item-cluster bootstrap 95% CIs. Each bootstrap resamples
    `eval_index_verb_context` items within verb/condition/metric, with all
    seed rows for a sampled item kept together through the item mean.
    """)
    return


@app.cell
def _():
    condition_plan = pl.DataFrame(
        [
            {
                "eval_construction": "transitive",
                "run_manipulation": "remove",
                "run_target": "e",
                "run_scope": "per-verb",
                "condition": "remove::e::per-verb",
                "condition_order": 0,
            },
            {
                "eval_construction": "transitive",
                "run_manipulation": "remove",
                "run_target": "p",
                "run_scope": "per-verb",
                "condition": "remove::p::per-verb",
                "condition_order": 1,
            },
            {
                "eval_construction": "causative",
                "run_manipulation": "substitute",
                "run_target": "p",
                "run_scope": "per-verb",
                "condition": "substitute::p::per-verb",
                "condition_order": 0,
            },
            {
                "eval_construction": "causative",
                "run_manipulation": "remove",
                "run_target": "p",
                "run_scope": "per-verb",
                "condition": "remove::p::per-verb",
                "condition_order": 1,
            },
        ]
    ).with_columns(
        pl.col("condition").replace(CONDITION_LABELS).alias("condition_label")
    )
    condition_plan
    return (condition_plan,)


@app.cell
def _(condition_plan):
    dataset_src = pl.scan_parquet(TESTSET_GLOB)
    expr_final_epoch = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max()

    dataset_filtered = (
        dataset_src.filter(
            pl.col("eval_has_context"),
            pl.col("eval_verb").is_in(VERB_TARGET_10),
            pl.col("eval_verb") == pl.col("run_target_verb"),
            expr_final_epoch,
        )
        .join(
            condition_plan.lazy(),
            on=[
                "eval_construction",
                "run_manipulation",
                "run_target",
                "run_scope",
            ],
            how="inner",
        )
        .select(
            "eval_construction",
            "condition",
            "condition_label",
            "condition_order",
            "run_seed",
            "eval_verb",
            "eval_index_verb_context",
            "result_nll",
            "result_slor",
        )
        .collect()
    )

    metric_df = (
        dataset_filtered.unpivot(
            on=list(METRICS.values()),
            index=[
                "eval_construction",
                "condition",
                "condition_label",
                "condition_order",
                "run_seed",
                "eval_verb",
                "eval_index_verb_context",
            ],
            variable_name="metric_col",
            value_name="metric_value",
        )
        .with_columns(
            pl.col("metric_col")
            .replace({v: k for k, v in METRICS.items()})
            .alias("metric")
        )
        .drop("metric_col")
    )
    return dataset_filtered, metric_df


@app.cell
def _(dataset_filtered):
    cell_summary = (
        dataset_filtered.group_by(
            "eval_construction",
            "condition",
            "condition_label",
        )
        .agg(
            pl.len().alias("rows"),
            pl.col("run_seed").n_unique().alias("seeds"),
            pl.col("eval_verb").n_unique().alias("verbs"),
            pl.col("eval_index_verb_context").n_unique().alias("items"),
            pl.col("result_nll").mean().alias("mean_nll"),
            pl.col("result_slor").mean().alias("mean_slor"),
        )
        .sort("eval_construction", "condition")
    )
    cell_summary
    return


@app.cell
def _():
    verb_order = (
        pl.read_parquet(VERB_STATS_PATH)
        .filter(pl.col("verb").is_in(VERB_TARGET_10))
        .sort("count_causative", descending=True)["verb"]
        .to_list()
    )
    verb_order
    return (verb_order,)


@app.cell
def _(metric_df):
    item_means = (
        metric_df.group_by(
            "eval_construction",
            "condition",
            "condition_label",
            "condition_order",
            "metric",
            "eval_verb",
            "eval_index_verb_context",
        )
        .agg(
            pl.col("metric_value").mean().alias("item_mean"),
            pl.col("run_seed").n_unique().alias("n_seed"),
        )
        .sort(
            "eval_construction",
            "metric",
            "eval_verb",
            "condition_order",
            "eval_index_verb_context",
        )
    )
    item_means
    return (item_means,)


@app.cell
def _(item_means):
    def bootstrap_mean_ci(
        values: np.ndarray,
        *,
        n_boot: int,
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        _values = np.asarray(values, dtype=float)
        _values = _values[np.isfinite(_values)]
        _n = len(_values)
        if _n == 0:
            return np.nan, np.nan, np.nan
        _mean = float(_values.mean())
        if _n == 1:
            return _mean, _mean, _mean

        _idx = rng.integers(0, _n, size=(n_boot, _n))
        _boot = _values[_idx].mean(axis=1)
        _low, _high = np.quantile(_boot, [0.025, 0.975])
        return _mean, float(_low), float(_high)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_rows = []

    for _row in (
        item_means.group_by(
            "eval_construction",
            "condition",
            "condition_label",
            "condition_order",
            "metric",
            "eval_verb",
        )
        .agg(
            pl.col("item_mean").alias("item_means"),
            pl.len().alias("n_item"),
            pl.col("n_seed").mean().alias("mean_seed_per_item"),
        )
        .iter_rows(named=True)
    ):
        _mean, _ci_low, _ci_high = bootstrap_mean_ci(
            np.asarray(_row["item_means"], dtype=float),
            n_boot=N_BOOT,
            rng=rng,
        )
        boot_rows.append(
            {
                "eval_construction": _row["eval_construction"],
                "condition": _row["condition"],
                "condition_label": _row["condition_label"],
                "condition_order": _row["condition_order"],
                "metric": _row["metric"],
                "eval_verb": _row["eval_verb"],
                "mean": _mean,
                "ci_low": _ci_low,
                "ci_high": _ci_high,
                "n_item": _row["n_item"],
                "mean_seed_per_item": _row["mean_seed_per_item"],
            }
        )

    boot_summary = pl.DataFrame(boot_rows).sort(
        "eval_construction",
        "metric",
        "eval_verb",
        "condition_order",
    )
    boot_summary
    return (boot_summary,)


@app.cell
def _(boot_summary, verb_order):
    def plot_bootstrapped_metrics():
        _fig, _axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        _axes_flat = _axes.ravel()
        _verb_frame = pl.DataFrame({"eval_verb": verb_order})
        _x = np.arange(len(verb_order))

        for _ax, (_construction, _metric) in zip(
            _axes_flat,
            PANEL_ORDER,
            strict=True,
        ):
            _sub = boot_summary.filter(
                (pl.col("eval_construction") == _construction)
                & (pl.col("metric") == _metric)
            )
            _conditions = (
                _sub.select(
                    "condition",
                    "condition_label",
                    "condition_order",
                )
                .unique()
                .sort("condition_order")
                .to_dicts()
            )
            _offsets = np.linspace(-0.13, 0.13, len(_conditions))

            for _offset, _condition_row in zip(
                _offsets,
                _conditions,
                strict=True,
            ):
                _condition = _condition_row["condition"]
                _ordered = _verb_frame.join(
                    _sub.filter(pl.col("condition") == _condition),
                    on="eval_verb",
                    how="left",
                )
                _mean = _ordered["mean"].to_numpy()
                _low = _ordered["ci_low"].to_numpy()
                _high = _ordered["ci_high"].to_numpy()
                _yerr = np.vstack([_mean - _low, _high - _mean])

                _ax.errorbar(
                    _x + _offset,
                    _mean,
                    yerr=_yerr,
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    markersize=5,
                    linewidth=1.2,
                    color=CONDITION_COLORS[_condition],
                    label=_condition_row["condition_label"],
                )

            _ax.set_title(f"{_construction} :: with_context [{_metric}]")
            _ax.set_ylabel(f"Mean {_metric}")
            _ax.grid(axis="y", linestyle=":", alpha=0.45)
            _ax.legend(frameon=False, fontsize=9)

        for _ax in _axes[-1, :]:
            _ax.set_xticks(_x)
            _ax.set_xticklabels(verb_order, rotation=45, ha="right")
            _ax.set_xlabel("Verb")

        _fig.suptitle(
            "Per-verb final-epoch NLL/SLOR with item-cluster bootstrap 95% CIs",
            y=1.02,
            fontsize=14,
        )
        _fig.tight_layout()
        return _fig

    boot_metric_fig = plot_bootstrapped_metrics()
    boot_metric_fig
    return


@app.cell
def _(boot_summary):
    boot_summary.select(
        "eval_construction",
        "metric",
        "eval_verb",
        "condition_label",
        "mean",
        "ci_low",
        "ci_high",
        "n_item",
        "mean_seed_per_item",
    ).sort("eval_construction", "metric", "eval_verb", "condition_label")
    return


if __name__ == "__main__":
    app.run()
