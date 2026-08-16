"""Abstract-vs-item-specific decomposition on trajectory data.

The removal-experiment decomposition in `lexical_vs_abstract_preemption.py`
isolates "abstract" (all-verb) vs "verb-specific" (per-verb) preemption via a
scope/target manipulation design. The trajectory experiment has no such
manipulation (every run is `manipulation="none"`), so that contrast can't be
replicated directly. Instead we use the shared-batch structure across
watched verbs: within a run, a single training batch can carry a causative
occurrence of verb X, of some other watched verb Y, both, or neither. That
gives a trajectory-native analogue of the same distinction:

  item_specific = verb X's own causative feature fired this step
  abstract      = X's own feature did not fire, but a co-watched verb Y's did
                   (construction-level exposure without X itself appearing)
  baseline      = neither (plain non-causative step for X)

Outcome and contrasts mirror `trajectory_step_type_effect.py`:

  nll_diff = NLL(post-update) - NLL(pre-update)

Contrasts (emmeans revpairwise over the 3-level `category` factor):
  item_specific - baseline
  abstract      - baseline
  item_specific - abstract

Direction: negative = that update type lowers transitive NLL more than the
comparison, i.e. increases overgeneralization.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE
from verb_overgen.exp.finale.util import despine
from verb_overgen.util.r_arrow_bridge import RScript

TRAJECTORY_DIR = Path("var/results/v2/trajectory")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

CATEGORY_LEVELS = ["baseline", "abstract", "item_specific"]

STEP_CONTRASTS = [
    ("early", "#4c78a8"),
    ("middle", "#f58518"),
    ("late", "#54a24b"),
]

_R_CONTRAST = RScript("""
library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom.mixed)
emm_options(lmer.df = "asymptotic")

entry <- function(df) {
    df <- df |>
        as_tibble() |>
        mutate(
            category = factor(category, levels = c("baseline", "abstract", "item_specific")),
            eval_verb = relevel(factor(eval_verb), ref = "go"),
            run_seed = factor(run_seed),
            log_step = log1p(step_effective)
        )

    model <- lmer(
        nll_diff ~ category * log_step * eval_verb + (1 | run_seed:eval_verb),
        weights = n_eval_items,
        data = df,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )

    fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) |>
        mutate(table = "fixed")

    marginal_contrast <- emmeans(model, revpairwise ~ category | eval_verb)$contrasts |>
        broom.mixed::tidy(conf.int = TRUE) |>
        mutate(table = "marginal_contrast")

    step_grid <- as.numeric(quantile(df$log_step, probs = c(0.1, 0.5, 0.9), names = FALSE))
    contrast_by_step <- emmeans(
        model,
        revpairwise ~ category | eval_verb * log_step,
        at = list(log_step = step_grid)
    )$contrasts |>
        broom.mixed::tidy(conf.int = TRUE) |>
        mutate(table = "contrast_by_step")

    bind_rows(fixed, marginal_contrast, contrast_by_step)
}
""")


def _causative_verbs_per_step(long: pl.LazyFrame) -> pl.LazyFrame:
    """Return {run_name, step_effective, causative_verbs: list[str]} for causative triggers."""
    return (
        long.filter(pl.col("step_type") == "causative")
        .select("run_name", "step_effective", "verb")
        .unique()
        .group_by("run_name", "step_effective")
        .agg(pl.col("verb").alias("causative_verbs"))
    )


def load_data(trajectory_dir: Path, bin_size: int, has_context: bool) -> pl.DataFrame:
    """Load trajectory rows, label each eval step by exposure category, and bin."""
    if bin_size < 1:
        raise ValueError("--bin-size must be >= 1")

    kind = "sentence_context_transitive" if has_context else "sentence_transitive"
    long = pl.scan_parquet(str(trajectory_dir / "*.parquet")).filter(
        pl.col("kind") == kind,
        pl.col("verb").is_in(VERB_TARGET_FINALE),
        pl.col("step_effective") > 0,
    )

    per_verb_step = long.group_by("verb", "step_effective", "index_sentence", "run_name").agg(
        pl.col("step_type").unique().item().alias("step_type"),
        pl.col("run_seed").unique().item().alias("run_seed"),
        pl.col("nll_last_sentence").filter(pl.col("phase") == "pre").item().alias("nll_pre"),
        pl.col("nll_last_sentence").filter(pl.col("phase") == "post").item().alias("nll_post"),
    )

    causative_lookup = _causative_verbs_per_step(long)

    df = per_verb_step.join(causative_lookup, on=["run_name", "step_effective"], how="left").with_columns(
        pl.col("causative_verbs").fill_null([]),
        (pl.col("nll_post") - pl.col("nll_pre")).alias("nll_diff"),
        (pl.col("step_effective") // bin_size * bin_size).alias("step_bin"),
    )

    # other_causative: some watched verb other than the row's own `verb` triggered causative
    # at this step_effective.
    df = df.with_columns(
        pl.struct(["verb", "causative_verbs"])
        .map_elements(
            lambda s: any(v != s["verb"] for v in s["causative_verbs"]),
            return_dtype=pl.Boolean,
        )
        .alias("other_causative")
    )

    df = df.with_columns(
        pl.when(pl.col("step_type") == "causative")
        .then(pl.lit("item_specific"))
        .when(pl.col("other_causative"))
        .then(pl.lit("abstract"))
        .otherwise(pl.lit("baseline"))
        .alias("category")
    )

    binned = (
        df.group_by("verb", "run_seed", "category", "step_bin")
        .agg(
            pl.col("nll_diff").mean().alias("nll_diff"),
            pl.col("step_effective").mean().alias("step_effective"),
            pl.col("index_sentence").n_unique().alias("n_eval_items"),
        )
        .rename({"verb": "eval_verb"})
        .select(
            "eval_verb",
            "run_seed",
            "category",
            "step_bin",
            "step_effective",
            "n_eval_items",
            "nll_diff",
        )
        .collect()
    )
    assert isinstance(binned, pl.DataFrame)
    return binned


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.select(
            "eval_verb",
            "run_seed",
            "category",
            "step_effective",
            "n_eval_items",
            "nll_diff",
        )
    )


def _verb_order(by: pl.Expr) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_FINALE))
    return (
        stats.with_columns((pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio"))
        .sort(by, descending=True)["verb"]
        .to_list()
    )


def plot_marginal_forest(contrasts: pl.DataFrame, out_path: Path) -> None:
    verbs = _verb_order(pl.col("count_causative"))
    marginal = contrasts.filter(pl.col("table") == "marginal_contrast")
    effects = marginal["contrast"].unique().to_list()

    fig, ax = plt.subplots(figsize=(7, max(4, len(verbs) * 0.55)))
    dodge = 0.7 / len(effects)
    y_pos = {v: i for i, v in enumerate(verbs)}
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b22222", "#74add1", "#e45756"]

    for idx, effect in enumerate(effects):
        sub = marginal.filter(pl.col("contrast") == effect)
        ordered = pl.DataFrame({"eval_verb": verbs}).join(sub, on="eval_verb", how="left")
        x = ordered["estimate"].to_numpy()
        xerr = ordered["std.error"].to_numpy() * 1.96
        y = np.array([y_pos[v] + (idx - (len(effects) - 1) / 2) * dodge for v in verbs])

        ax.errorbar(
            x=x,
            y=y,
            xerr=xerr,
            fmt="o",
            capsize=3,
            color=colors[idx % len(colors)],
            markersize=5,
            linewidth=1.2,
            label=effect,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels(verbs)
    ax.set_xlabel("Contrast on delta NLL (category A - category B)")
    ax.set_ylabel("Verb")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(title="Contrast", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    despine(ax)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _with_step_labels(contrasts: pl.DataFrame) -> pl.DataFrame:
    by_step = contrasts.filter(pl.col("table") == "contrast_by_step")
    steps = sorted(by_step["log_step"].drop_nulls().unique().to_list())
    parts = []
    for (label, _color), step in zip(STEP_CONTRASTS, steps, strict=True):
        parts.append(
            by_step.filter((pl.col("log_step") - step).abs() < 1e-8).with_columns(
                pl.lit(label).alias("step_label"),
                pl.lit(float(np.expm1(step))).alias("step_effective_at"),
            )
        )
    return pl.concat(parts)


def print_summary(contrasts: pl.DataFrame, context_tag: str = "with_context") -> None:
    print(f"\n{'=' * 64}")
    print("  Trajectory abstract-vs-item-specific effect")
    print(f"  Eval: transitive :: {context_tag}")
    print("  Categories: item_specific (own causative) / abstract (other-verb causative,")
    print("              same step) / baseline (neither)")
    print("  Direction: negative = that category lowers transitive NLL more")
    print(f"{'=' * 64}")

    marginal = contrasts.filter(pl.col("table") == "marginal_contrast")
    by_step = _with_step_labels(contrasts)
    with pl.Config(tbl_rows=-1):
        print("\nMarginal over trajectory:")
        print(
            marginal.select("contrast", "eval_verb", "estimate", "std.error", "statistic", "adj.p.value").sort(
                "contrast", "eval_verb"
            )
        )
        print("\nAt early/middle/late trajectory points:")
        print(
            by_step.select(
                "step_label",
                "contrast",
                "eval_verb",
                "step_effective_at",
                "estimate",
                "std.error",
                "statistic",
                "adj.p.value",
            ).sort("step_label", "contrast", "eval_verb")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=TRAJECTORY_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--bin-size",
        type=int,
        default=50,
        help="Training-step bin width before fitting the mixed model (default: %(default)s)",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Use no_context eval condition instead of with_context (default: with_context)",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    has_context = not args.no_context
    context_tag = "no_context" if args.no_context else "with_context"

    print("Loading trajectory data...")
    df = load_data(args.trajectory_dir, args.bin_size, has_context)
    data_summary = "  {:,} binned rows  |  {} verbs  |  {} seeds  |  bin size: {}".format(
        len(df),
        df["eval_verb"].n_unique(),
        df["run_seed"].n_unique(),
        args.bin_size,
    )
    print(data_summary)
    print("  category counts:", df["category"].value_counts().sort("category").to_dicts())

    print("Fitting single mixed model across verbs...")
    contrasts = run_contrast(df)
    print_summary(contrasts, context_tag)

    plot_marginal_forest(
        contrasts,
        args.figure_dir / f"20260714.trajectory-abstract-vs-item-specific_{context_tag}-transitive.pdf",
    )


if __name__ == "__main__":
    main()
