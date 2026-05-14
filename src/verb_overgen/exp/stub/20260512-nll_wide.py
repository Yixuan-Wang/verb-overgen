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
    import seaborn as sns
    import scipy.stats as stats

    from verb_overgen.common.features import VERB_TARGET_10

    RUNS_OF_INTEREST = [
        ("none", "", ""),
        ("remove", "p", "all-verb"),
        ("remove", "p", "except-verb"),
        ("remove", "p", "per-verb"),
        ("remove", "e", "per-verb"),
    ]

    RUN_LABELS = {
        ("remove", "p", "all-verb"): "remove :: p :: all-verb",
        ("remove", "p", "except-verb"): "remove :: p :: except-verb",
        ("remove", "p", "per-verb"): "remove :: p :: per-verb",
        ("remove", "e", "per-verb"): "remove :: e :: per-verb",
        ("none", "", ""): "none",
    }


@app.cell
def _():
    ui_verb = mo.ui.dropdown(
        options=VERB_TARGET_10,
        value=VERB_TARGET_10[0],
        label="Verb",
    )
    ui_verb
    return (ui_verb,)


@app.cell
def _(ui_verb):
    selected_verb = ui_verb.value
    return (selected_verb,)


@app.cell
def _(selected_verb):
    dataset_src = pl.scan_parquet("var/results/v2/testset/*.parquet")

    expr_epoch_final = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max()
    expr_verb_equal = (pl.col("run_scope").is_in(["all-verb", ""])) | (
        pl.col("run_target_verb") == pl.col("eval_verb")
    )

    runs_df = pl.DataFrame(
        RUNS_OF_INTEREST,
        schema=["run_manipulation", "run_target", "run_scope"],
        orient="row",
    )

    dataset_filtered = (
        dataset_src.filter(
            pl.col("eval_verb") == selected_verb,
            pl.col("eval_construction") == "transitive",
            pl.col("eval_has_context") == True,
            expr_epoch_final,
            expr_verb_equal,
        )
        .join(runs_df.lazy(), on=["run_manipulation", "run_target", "run_scope"], how="inner")
        .collect()
    )

    dataset_labeled = dataset_filtered.with_columns(
        pl.concat_list(
            pl.col("run_manipulation"),
            pl.col("run_target"),
            pl.col("run_scope"),
        )
        .list.filter(pl.element() != "")
        .list.join(" :: ")
        .alias("run_label")
    )
    return (dataset_labeled,)


@app.cell
def _(dataset_labeled, selected_verb):
    wide_df = (
        dataset_labeled
        .group_by("eval_index_sentence", "run_label", "run_seed")
        .agg(pl.col("result_nll").mean().alias("nll"))
        .pivot(on="run_label", index=["run_seed", "eval_index_sentence"], values="nll")
        .sort("eval_index_sentence")
    )

    col_order = ["run_seed", "eval_index_sentence"] + [
        v for v in RUN_LABELS.values() if v in wide_df.columns
    ]
    wide_df = wide_df.select([c for c in col_order if c in wide_df.columns])

    mo.vstack([
        mo.md(f"**{selected_verb}** — transitive :: with_context, NLL"),
        wide_df,
    ])
    return (wide_df,)


@app.cell
def _(wide_df):
    def _ratio(expr: pl.Expr):
        return expr.sum() / expr.len()

    _df = wide_df.select(
        "run_seed",
        (pl.col("remove :: e :: per-verb") - pl.col("remove :: p :: per-verb")).alias("lexical-with-abstract-alt"),
        (pl.col("none") - pl.col("remove :: p :: per-verb")).alias("lexical-with-abstract"),
        (pl.col("remove :: p :: except-verb") - pl.col("remove :: p :: all-verb")).alias("lexical-no-abstract"),
    ).with_columns(
        pl.col("lexical-with-abstract").sub(pl.col("lexical-no-abstract")).alias("gap")
    )

    _df
    return


if __name__ == "__main__":
    app.run()
