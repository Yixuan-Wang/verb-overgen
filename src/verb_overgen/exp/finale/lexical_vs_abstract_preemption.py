"""Preemption-vs-entrenchment decomposition (DiD against the matched `e` arm).

For each (scope ∈ {all-verb, per-verb, except-verb}) cell, both the
`remove::p` and the count-matched `remove::e` arms are loaded. Each
preemption contrast is then computed as a difference-in-differences against
its `e` twin, isolating the preemption-specific effect from the generic
entrenchment / data-volume effect of removing N verb-instances.

Final-epoch `transitive :: with_context` or `transitive :: no_context`. DiD
contrasts per verb:

  abstract                    = all_e − all_p
  item_specific_given_indirect = per_e − per_p
  indirect_given_item_specific = except_e − except_p    (mirror of item_specific_given_indirect)

Direction: NLL ↑ / SLOR ↓ = more preemption-specific effect beyond
entrenchment. (Equivalently, the sign convention reverses the original
per-verb contrast's `p − e`: positive here = `e > p`, the
preemption-hypothesis-supporting direction.)
"""
from collections.abc import Iterable

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE, VERB_TARGET_CANDIDATE_V2_ALTERNATING
from verb_overgen.exp.finale.util import ACL_COL_WIDTH, apply_acl_style, despine
from verb_overgen.util.r_arrow_bridge import RScript

EVAL_VERBS = [*VERB_TARGET_FINALE, ] # "stop", "break"]

TESTSET_DIRS = [
    Path("var/results/v2/testset_0818_wordlevel"),
    Path("var/results/v2/testset_0818_wordlevel_alternating"),
]
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

METRICS: dict[str, tuple[str, str]] = {
    "nll": ("result_nll", "NLL"),
    # "slor": ("result_slor", "SLOR"),
}

# (effect key — must match R contrast names, display label, color)
CONTRASTS = [
    # ("contrast_1", "contrast 1", "#b22222", "s"),
    # ("contrast_2", "contrast 2", "#f4a0a0", "D"),
    ("abstract", "Abstract", "#2d3748", "o"),
    ("item_specific_given_indirect", "Verb-specific", "#74add1", "s"),
    # ("indirect_given_item_specific", "Indirect", "#f46d43", "^"),
]

_R_CONTRAST = RScript("""
library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

entry <- function(df) {
    df <- df |>
        as_tibble() |>
        mutate(
            cell = factor(
                paste(target, scope, sep = "_"),
                levels = c("p_all-verb",    "e_all-verb",
                           "p_per-verb",    "e_per-verb",
                           "p_except-verb", "e_except-verb")
            ),
            eval_verb = relevel(factor(eval_verb), ref = "go")
        )

    model <- lmer(
        metric ~ cell * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    # Cell order: 1=p_all, 2=e_all, 3=p_per, 4=e_per, 5=p_except, 6=e_except.
    # Positive contrast = preemption-specific effect beyond entrenchment.
    model |>
        emmeans(~ cell | eval_verb) |>
        contrast(method = list(
            abstract                    = c(-1, +1,  0,  0,  0,  0),
            item_specific_given_indirect = c( 0,  0, -1, +1,  0,  0)
            # indirect_given_item_specific = c( 0,  0,  0,  0, -1, +1)
        ), adjust = "mvt") |>
        tidy()
}
""")


def load_data(testset_dir: Path | Iterable[Path], has_context: bool) -> pl.DataFrame:
    dirs = (testset_dir,) if isinstance(testset_dir, Path) else tuple(testset_dir)
    src = pl.concat([pl.scan_parquet(d / "*.parquet") for d in dirs], how="vertical")

    # Per-run final epoch: testset dirs may have been collected with different
    # checkpoint cadences, so a global max would drop shorter-logged runs.
    final_epoch = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max().over("run_name")
    common_cols = ["run_seed", "eval_verb", "eval_index_verb_context", "result_nll"] # , "result_slor"]
    common_filter = [
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context") == has_context,
        pl.col("eval_verb").is_in(EVAL_VERBS),
        final_epoch,
    ]

    def remove_arm(scope: str, target: str) -> pl.LazyFrame:
        filters = [
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == scope,
            pl.col("run_target") == target,
            *common_filter,
        ]
        if scope in ("per-verb", "except-verb"):
            filters.append(pl.col("eval_verb") == pl.col("run_target_verb"))
        return (
            src.filter(*filters)
            .select(common_cols)
            .with_columns(
                pl.lit(scope).alias("scope"),
                pl.lit(target).alias("target"),
            )
        )

    arms = [remove_arm(scope, target) for scope in ("all-verb", "per-verb", "except-verb") for target in ("p", "e")]
    return pl.concat([a.collect() for a in arms])


def run_contrast(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({metric_col: "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    ).rename({"contrast": "effect"})


def _verb_order(by: pl.Expr) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(EVAL_VERBS))
    return (
        stats.with_columns((pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio"))
        .sort(by, descending=True)["verb"]
        .to_list()
    )


def plot_forest(contrasts: pl.DataFrame, metric_label: str, out_path: Path) -> None:
    verbs = _verb_order(pl.col("count_causative"))
    n_effects = len(CONTRASTS)
    dodge = 0.7 / n_effects
    y_pos = {v: i for i, v in enumerate(verbs)}

    height = max(2.8, len(verbs) * 0.35)
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, height))

    for idx, (effect_key, effect_label, color, marker) in enumerate(CONTRASTS):
        sub = contrasts.filter(pl.col("effect") == effect_key)
        ordered = pl.DataFrame({"eval_verb": verbs}).join(sub, on="eval_verb", how="left")
        x = ordered["estimate"].to_numpy()
        xerr = ordered["std.error"].to_numpy() * 1.96
        y = np.array([y_pos[v] + (idx - (n_effects - 1) / 2) * dodge for v in verbs])

        ax.errorbar(
            x=x,
            y=y,
            xerr=xerr,
            fmt=marker,
            capsize=2,
            color=color,
            markersize=3,
            linewidth=1,
            label=effect_label,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels([fr"$\textit{{{verb}}}$" for verb in verbs])

    arrow_better = r"$\uparrow$" if metric_label == "NLL" else r"$\downarrow$"
    ax.set_xlabel(
        rf"{metric_label} Contrast (\textsc{{keep-p}} $-$ \textsc{{remove-p}}) {arrow_better}"
    )
    ax.set_ylabel("Verb")
    # ticks = ax.get_xticks()
    ticks = np.linspace(-0.18, 0.18, num=7)
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0] - 0.02, ticks[-1] + 0.02)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3, frameon=False)
    despine(ax)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, metric_label: str, context_tag: str) -> None:
    direction = "↑" if metric_label == "NLL" else "↓"
    print(f"\n{'=' * 64}")
    print(f"  {metric_label}  —  overall preemption effect: verb-specific/indirect/abstract decomposition")
    print(f"  Eval: transitive :: {context_tag}")
    print(f"  Direction: {direction} = preemption-specific effect beyond entrenchment")
    print(f"{'=' * 64}")
    with pl.Config(tbl_rows=-1):
        print(
            contrasts.select("effect", "eval_verb", "estimate", "std.error", "statistic", "adj.p.value").sort(
                "effect", "eval_verb"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testset-dir",
        type=Path,
        nargs="+",
        default=TESTSET_DIRS,
        help="One or more testset result dirs; their parquet files are concatenated.",
    )
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Use no_context eval condition instead of with_context (default: with_context)",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    has_context = not args.no_context
    context_tag = "no_context" if args.no_context else "with_context"

    testset_dirs = args.testset_dir if isinstance(args.testset_dir, list) else [args.testset_dir]
    testset_tag = "+".join(d.name for d in testset_dirs)

    print(f"Loading testset data from {[str(d) for d in testset_dirs]} ...")
    df = load_data(testset_dirs, has_context)
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    for metric_key, (metric_col, metric_label) in METRICS.items():
        contrasts = run_contrast(df, metric_col)
        print_summary(contrasts, metric_label, context_tag)
        out_path = (
            args.figure_dir
            / f"20260823.{testset_tag}_{context_tag}_{metric_key}.pdf"
        )
        plot_forest(contrasts, metric_label, out_path)


if __name__ == "__main__":
    main()
