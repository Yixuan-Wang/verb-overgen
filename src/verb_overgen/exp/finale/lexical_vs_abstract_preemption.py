"""Preemption-vs-entrenchment decomposition (DiD against the matched `e` arm).

For each (scope ∈ {all-verb, per-verb, except-verb}) cell, both the
`remove::p` and the count-matched `remove::e` arms are loaded. Each
preemption contrast is then computed as a difference-in-differences against
its `e` twin, isolating the preemption-specific effect from the generic
entrenchment / data-volume effect of removing N verb-instances.

Final-epoch `transitive :: with_context` or `transitive :: no_context`. DiD
contrasts per verb:

  total              = all_e − all_p          (`none` cancels in DiD)
  lex_given_abstract = per_e − per_p          (`none` cancels in DiD)
  abstract_given_lex = except_e − except_p    (mirror of lex_given_abstract)

Direction: NLL ↑ / SLOR ↓ = more preemption-specific effect beyond
entrenchment. (Equivalently, the sign convention reverses the original
per-verb contrast's `p − e`: positive here = `e > p`, the
preemption-hypothesis-supporting direction.)
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE
from verb_overgen.util.r_arrow_bridge import RScript

TESTSET_DIR = Path("var/results/v2/testset")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

# ACL figure style (acl.sty two-column layout)
ACL_COL_WIDTH = 3.18  # \columnwidth, inches

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METRICS: dict[str, tuple[str, str]] = {
    "nll": ("result_nll", "NLL"),
    "slor": ("result_slor", "SLOR"),
}

# (effect key — must match R contrast names, display label, color)
CONTRASTS = [
    # ("contrast_1", "contrast 1", "#b22222", "s"),
    # ("contrast_2", "contrast 2", "#f4a0a0", "D"),
    ("total", "Total", "black", "o"),
    ("lex_given_abstract", "Lexical", "#1f77b4", "s"),
    ("abstract_given_lex", "Abstract", "#2ca02c", "^"),
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
            total              = c(-1, +1,  0,  0,  0,  0),
            lex_given_abstract = c( 0,  0, -1, +1,  0,  0),
            abstract_given_lex = c( 0,  0,  0,  0, -1, +1)
        ), adjust = "mvt") |>
        tidy()
}
""")


def load_data(testset_dir: Path, has_context: bool) -> pl.DataFrame:
    src = pl.scan_parquet(testset_dir / "*.parquet")
    final_epoch = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max()
    common_cols = ["run_seed", "eval_verb", "eval_index_verb_context", "result_nll", "result_slor"]
    common_filter = [
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context") == has_context,
        pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
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
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_FINALE))
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
    ax.set_yticklabels(verbs)

    arrow_better = r"$\uparrow$" if metric_label == "NLL" else r"$\downarrow$"
    ax.set_xlabel(
        rf"{metric_label} Contrast (\textsc{{keep-p}} $-$ \textsc{{remove-p}}) {arrow_better}"
    )
    ax.set_ylabel("Verb")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0] - 0.02, ticks[-1] + 0.02)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, metric_label: str, context_tag: str) -> None:
    direction = "↑" if metric_label == "NLL" else "↓"
    print(f"\n{'=' * 64}")
    print(f"  {metric_label}  —  overall preemption effect: lexical/abstract decomposition")
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
    parser.add_argument("--testset-dir", type=Path, default=TESTSET_DIR)
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

    print("Loading testset data...")
    df = load_data(args.testset_dir, has_context)
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    for metric_key, (metric_col, metric_label) in METRICS.items():
        contrasts = run_contrast(df, metric_col)
        print_summary(contrasts, metric_label, context_tag)
        out_path = args.figure_dir / f"20260513.v3.lex-vs-abstract-preemption-did_{context_tag}_{metric_key}.pdf"
        plot_forest(contrasts, metric_label, out_path)


if __name__ == "__main__":
    main()
