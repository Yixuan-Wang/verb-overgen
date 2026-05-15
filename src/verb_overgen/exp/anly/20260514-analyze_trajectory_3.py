# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import os

    if "R_HOME" not in os.environ:
        import subprocess
        import json

        _direnv_env = subprocess.check_output(
            ["direnv", "export", "json"], text=True
        )
        if _direnv_env:
            os.environ.update(json.loads(_direnv_env))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    import scipy.stats as stats

    from verb_overgen.util.r_arrow_bridge import RScript

    os.chdir(os.environ["PWD"])


@app.cell
def _():
    dataset_step = pl.scan_parquet("./var/results/v2/trajectory/*.parquet")
    # dataset_step = dataset_step.with_columns(
    #     pl.when(pl.col("phase") == "pre").then(pl.col("step")).otherwise(pl.col("step") - 1).alias("step_effective")
    # )
    return (dataset_step,)


@app.cell
def _():
    dataset_testset = pl.read_parquet(
        "./data/testset/output/testset_gemini.20260402T2045.parquet"
    )
    dataset_testset_info = dataset_testset.select(
        "index_sentence",
        "verb",
        "index_verb_context",
        "sentence",
    )
    return (dataset_testset_info,)


@app.cell
def _(dataset_step):
    dataset_step_verb: pl.DataFrame = (  # type: ignore
        dataset_step.filter(pl.col("verb") == "go").collect()
    )
    dataset_step_verb
    return (dataset_step_verb,)


@app.cell
def _(dataset_step_verb: pl.DataFrame, dataset_testset_info):
    dataset_step_verb_diff = (
        dataset_step_verb.group_by("step_effective", "index_sentence", "run_name")
        .agg(
            pl.col("kind").unique().item().alias("kind"),
            pl.col("step_type").unique().item().alias("step_type"),
            pl.col("run_seed").unique().item().alias("run_seed"),
            pl.col("^feature_count_.*$").unique().item(),
            pl.col("nll_last_sentence")
            .filter(pl.col("phase") == "pre")
            .item()
            .alias("nll_pre"),
            pl.col("nll_last_sentence")
            .filter(pl.col("phase") == "post")
            .item()
            .alias("nll_post"),
        )
        .sort("step_effective", "index_sentence")
        .with_columns((pl.col("nll_post") - pl.col("nll_pre")).alias("nll_diff"))
        .join(dataset_testset_info, on="index_sentence", validate="m:1")
    )

    dataset_step_verb_diff
    return (dataset_step_verb_diff,)


@app.cell
def _(dataset_step_verb_diff):
    _df = (
        dataset_step_verb_diff.lazy()
        .filter(
            pl.col("kind") == "sentence_context_transitive",
            # pl.col("index_verb_context") == 24,
        )
        .group_by("run_seed", "step_effective")
        .agg(pl.col("step_type").unique().item(), pl.col("nll_diff").mean())
        .collect()
        .sort("step_effective")
    )

    # sns.scatterplot(
    #     _df,
    #     x="step_effective",
    #     y="nll_diff",
    #     hue="step_type",
    #     palette={
    #         "causative": "orange",
    #         "non_causative": "grey",
    #     }
    # )
    sns.kdeplot(
        _df,
        hue="step_type",
        y="nll_diff",
        palette={
            "causative": "orange",
            "non_causative": "grey",
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare `causative` against `non_causative` as the update type.

    Outcome is `nll_diff = nll_post - nll_pre`, so a negative value means the
    update lowered NLL for the held-out sentence. The treatment contrast below
    uses `non_causative` as the reference. For `sentence_context_transitive`,
    `step_typecausative < 0` means causative updates make contextual transitive
    overgeneralizations more probable than non-causative updates; `> 0` means
    they suppress those sentences relative to non-causative updates.
    """)
    return


@app.cell
def _():
    fit_step_type_effect = RScript("""
    library(arrow); library(dplyr); library(lme4); library(lmerTest); library(broom.mixed); library(emmeans)
    emm_options(lmer.df = "asymptotic")

    entry <- function(df) {
        df <- df %>%
            as_tibble() %>%
            mutate(
                step_type = relevel(factor(step_type), ref = "non_causative"),
                log_step = log1p(step_effective),
                run_seed = factor(run_seed),
                index_verb_context = factor(index_verb_context)
            )

        model <- lmer(
            nll_diff ~ step_type * log_step
                + (1 | run_seed:index_verb_context),
            data = df,
            control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
        )

        fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) %>%
            mutate(table = "fixed")

        marginal_contrast <- emmeans(model, revpairwise ~ step_type)$contrasts %>%
            broom.mixed::tidy(conf.int = TRUE) %>%
            mutate(table = "marginal_contrast")

        step_grid <- as.numeric(quantile(df$log_step, probs = c(0.1, 0.5, 0.9), names = FALSE))
        contrast_by_step <- emmeans(model, revpairwise ~ step_type | log_step, at = list(log_step = step_grid))$contrasts %>%
            broom.mixed::tidy(conf.int = TRUE) %>%
            mutate(table = "contrast_by_step")

        bind_rows(fixed, marginal_contrast, contrast_by_step)
    }
    """)
    return (fit_step_type_effect,)


@app.cell
def _(dataset_step_verb_diff, fit_step_type_effect):
    step_type_effect_context_transitive = (
        dataset_step_verb_diff
        .filter(
            pl.col("kind") == "sentence_context_transitive",
            pl.col("step_effective") > 0,
        )
        .select(
            "nll_diff",
            "step_effective",
            "step_type",
            "run_seed",
            "index_verb_context",
        )
        .pipe(fit_step_type_effect.call_table)
    )
    step_type_effect_context_transitive
    return


@app.cell
def _(dataset_step_verb_diff):
    step_type_bin_contrast = (
        dataset_step_verb_diff
        .filter(pl.col("kind") == "sentence_context_transitive")
        .with_columns((pl.col("step_effective") // 100 * 100).alias("step_bin"))
        .group_by("run_seed", "step_bin", "step_type")
        .agg(pl.col("nll_diff").mean().alias("nll_diff_mean"))
        .pivot(
            on="step_type",
            index=["run_seed", "step_bin"],
            values="nll_diff_mean",
        )
        .drop_nulls(["causative", "non_causative"])
        .with_columns(
            (pl.col("causative") - pl.col("non_causative")).alias(
                "nll_diff_causative_minus_non"
            )
        )
    )

    _contrast = step_type_bin_contrast["nll_diff_causative_minus_non"].to_numpy()
    _test = stats.ttest_1samp(_contrast, popmean=0.0)
    step_type_bin_summary = pl.DataFrame(
        {
            "n_bins": [len(_contrast)],
            "mean_contrast": [float(np.mean(_contrast))],
            "std_contrast": [float(np.std(_contrast, ddof=1))],
            "t_statistic": [float(_test.statistic)],
            "p_value": [float(_test.pvalue)],
        }
    )

    step_type_bin_summary
    return


if __name__ == "__main__":
    app.run()
