"""Test whether causative updates differ from non-causative updates.

Single mixed-effects model across verbs, using trajectory data rather than
final-epoch checkpoint data.  The primary eval condition is
`transitive :: with_context`, because those are the overgeneralization items.

Outcome:
  nll_diff = NLL(post-update) - NLL(pre-update)

Contrast:
  causative - non_causative

Direction:
  contrast < 0 means causative updates lower transitive NLL more than
  non-causative updates, making the overgeneralization sentences more probable.
  contrast > 0 means causative updates suppress those sentences relative to
  non-causative updates.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_10
from verb_overgen.util.r_arrow_bridge import RScript

TRAJECTORY_DIR = Path("var/results/v2/trajectory")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

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
            step_type = factor(step_type, levels = c("non_causative", "causative")),
            eval_verb = relevel(factor(eval_verb), ref = "go"),
            run_seed = factor(run_seed),
            log_step = log1p(step_effective)
        )

    model <- lmer(
        nll_diff ~ step_type * log_step * eval_verb + (1 | run_seed:eval_verb),
        weights = n_eval_items,
        data = df,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )

    fixed <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE) |>
        mutate(table = "fixed")

    marginal_contrast <- emmeans(model, revpairwise ~ step_type | eval_verb)$contrasts |>
        broom.mixed::tidy(conf.int = TRUE) |>
        mutate(table = "marginal_contrast")

    step_grid <- as.numeric(quantile(df$log_step, probs = c(0.1, 0.5, 0.9), names = FALSE))
    contrast_by_step <- emmeans(
        model,
        revpairwise ~ step_type | eval_verb * log_step,
        at = list(log_step = step_grid)
    )$contrasts |>
        broom.mixed::tidy(conf.int = TRUE) |>
        mutate(table = "contrast_by_step")

    bind_rows(fixed, marginal_contrast, contrast_by_step)
}
""")


def load_data(trajectory_dir: Path, bin_size: int, has_context: bool) -> pl.DataFrame:
    """Load trajectory rows and aggregate to per-seed/verb/update-bin means."""
    if bin_size < 1:
        raise ValueError("--bin-size must be >= 1")

    kind = "sentence_context_transitive" if has_context else "sentence_transitive"
    df = (
        pl.scan_parquet(str(trajectory_dir / "*.parquet"))
        .filter(
            pl.col("kind") == kind,
            pl.col("verb").is_in(VERB_TARGET_10),
            pl.col("step_effective") > 0,
        )
        .group_by("verb", "step_effective", "index_sentence", "run_name")
        .agg(
            pl.col("step_type").unique().item().alias("step_type"),
            pl.col("run_seed").unique().item().alias("run_seed"),
            pl.col("nll_last_sentence")
            .filter(pl.col("phase") == "pre")
            .item()
            .alias("nll_pre"),
            pl.col("nll_last_sentence")
            .filter(pl.col("phase") == "post")
            .item()
            .alias("nll_post"),
        )
        .with_columns(
            (pl.col("nll_post") - pl.col("nll_pre")).alias("nll_diff"),
            (pl.col("step_effective") // bin_size * bin_size).alias("step_bin"),
        )
        .group_by("verb", "run_seed", "step_type", "step_bin")
        .agg(
            pl.col("nll_diff").mean().alias("nll_diff"),
            pl.col("step_effective").mean().alias("step_effective"),
            pl.col("index_sentence").n_unique().alias("n_eval_items"),
        )
        .rename({"verb": "eval_verb"})
        .select(
            "eval_verb",
            "run_seed",
            "step_type",
            "step_bin",
            "step_effective",
            "n_eval_items",
            "nll_diff",
        )
        .collect()
    )
    assert isinstance(df, pl.DataFrame)
    return df


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.select(
            "eval_verb",
            "run_seed",
            "step_type",
            "step_effective",
            "n_eval_items",
            "nll_diff",
        )
    )


def _verb_order(by: pl.Expr) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_10))
    return (
        stats.with_columns(
            (pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio")
        )
        .sort(by, descending=True)["verb"]
        .to_list()
    )


def plot_marginal_forest(contrasts: pl.DataFrame, out_path: Path) -> None:
    verbs = _verb_order(pl.col("count_causative"))
    marginal = contrasts.filter(pl.col("table") == "marginal_contrast")
    ordered = pl.DataFrame({"eval_verb": verbs}).join(marginal, on="eval_verb", how="left")

    fig, ax = plt.subplots(figsize=(7, max(4, len(verbs) * 0.55)))
    x = ordered["estimate"].to_numpy()
    xerr = ordered["std.error"].to_numpy() * 1.96

    ax.errorbar(
        x=x,
        y=list(range(len(verbs))),
        xerr=xerr,
        fmt="o",
        capsize=4,
        color="black",
        markersize=6,
        linewidth=1.5,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels(verbs)
    ax.set_xlabel("Contrast: causative - non-causative update effect on delta NLL")
    ax.set_ylabel("Verb")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
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


def plot_step_forest(contrasts: pl.DataFrame, out_path: Path) -> None:
    verbs = _verb_order(pl.col("count_causative"))
    by_step = _with_step_labels(contrasts)
    y_pos = {verb: idx for idx, verb in enumerate(verbs)}
    dodge = 0.7 / len(STEP_CONTRASTS)

    fig, ax = plt.subplots(figsize=(8, max(4, len(verbs) * 0.65)))
    for idx, (label, color) in enumerate(STEP_CONTRASTS):
        sub = by_step.filter(pl.col("step_label") == label)
        ordered = pl.DataFrame({"eval_verb": verbs}).join(sub, on="eval_verb", how="left")
        x = ordered["estimate"].to_numpy()
        xerr = ordered["std.error"].to_numpy() * 1.96
        y = np.array([y_pos[verb] + (idx - (len(STEP_CONTRASTS) - 1) / 2) * dodge for verb in verbs])
        step_at = ordered["step_effective_at"].drop_nulls()[0]

        ax.errorbar(
            x=x,
            y=y,
            xerr=xerr,
            fmt="o",
            capsize=3,
            color=color,
            markersize=5,
            linewidth=1.2,
            label=f"{label} (step approx. {step_at:,.0f})",
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels(verbs)
    ax.set_xlabel("Contrast: causative - non-causative update effect on delta NLL")
    ax.set_ylabel("Verb")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(title="Trajectory point", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, context_tag: str = "with_context") -> None:
    print(f"\n{'=' * 64}")
    print("  Trajectory step-type effect")
    print(f"  Eval: transitive :: {context_tag}")
    print("  Contrast: causative - non_causative on delta NLL")
    print("  Direction: negative = causative updates lower transitive NLL more")
    print(f"{'=' * 64}")

    marginal = contrasts.filter(pl.col("table") == "marginal_contrast")
    by_step = _with_step_labels(contrasts)
    with pl.Config(tbl_rows=-1):
        print("\nMarginal over trajectory:")
        print(
            marginal.select("eval_verb", "estimate", "std.error", "statistic", "p.value")
            .sort("eval_verb")
        )
        print("\nAt early/middle/late trajectory points:")
        print(
            by_step.select(
                "step_label",
                "eval_verb",
                "step_effective_at",
                "estimate",
                "std.error",
                "statistic",
                "p.value",
            ).sort("step_label", "eval_verb")
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
    context_tag = "no-context" if args.no_context else "with-context"

    print("Loading trajectory data...")
    df = load_data(args.trajectory_dir, args.bin_size, has_context)
    data_summary = "  {:,} binned rows  |  {} verbs  |  {} seeds  |  bin size: {}".format(
        len(df),
        df["eval_verb"].n_unique(),
        df["run_seed"].n_unique(),
        args.bin_size,
    )
    print(data_summary)

    print("Fitting single mixed model across verbs...")
    contrasts = run_contrast(df)
    print_summary(contrasts, context_tag)

    plot_marginal_forest(
        contrasts,
        args.figure_dir / f"20260514.trajectory-step-type-effect_{context_tag}-transitive.pdf",
    )
    plot_step_forest(
        contrasts,
        args.figure_dir / f"20260514.trajectory-step-type-effect_by-step_{context_tag}-transitive.pdf",
    )


if __name__ == "__main__":
    main()
