"""Preemption-vs-entrenchment decomposition (DiD against the matched `e` arm).

For each (scope ∈ {all-verb, per-verb, except-verb}) cell, both the
`remove::p` and the count-matched `remove::e` arms are loaded. Each
preemption contrast is then computed as a difference-in-differences against
its `e` twin, isolating the preemption-specific effect from the generic
entrenchment / data-volume effect of removing N verb-instances.

Final-epoch `transitive :: with_context`. DiD contrasts per verb:

  lexical_alone      = (except − all)_p − (except − all)_e
  abstract_alone     = (per    − all)_p − (per    − all)_e
  total              = all_e − all_p          (`none` cancels in DiD)
  lex_given_abstract = per_e − per_p          (`none` cancels in DiD)
  abstract_given_lex = except_e − except_p    (mirror of lex_given_abstract)

Direction: NLL ↑ / SLOR ↓ = more preemption-specific effect beyond
entrenchment. (Equivalently, the sign convention reverses Exp 1's `p − e`:
positive here = `e > p`, the preemption-hypothesis-supporting direction.)
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_10
from verb_overgen.util.r_arrow_bridge import RScript

TESTSET_DIR = Path("var/results/v2/testset")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

METRICS: dict[str, tuple[str, str]] = {
    "nll": ("result_nll", "NLL"),
    "slor": ("result_slor", "SLOR"),
}

# (effect key — must match R contrast names, display label, color)
CONTRASTS = [
    ("lexical_alone",      "lexical (alone)",      "#b22222"),
    ("abstract_alone",     "abstract (alone)",     "#f4a0a0"),
    ("total",              "total",                "black"),
    ("lex_given_abstract", "lex | abstract on",    "#1f77b4"),
    ("abstract_given_lex", "abstract | lex on",    "#2ca02c"),
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
            lexical_alone      = c(-1, +1,  0,  0, +1, -1),
            abstract_alone     = c(-1, +1, +1, -1,  0,  0),
            total              = c(-1, +1,  0,  0,  0,  0),
            lex_given_abstract = c( 0,  0, -1, +1,  0,  0),
            abstract_given_lex = c( 0,  0,  0,  0, -1, +1)
        ), adjust = "mvt") |>
        tidy()
}
""")


def load_data() -> pl.DataFrame:
    src = pl.scan_parquet(TESTSET_DIR / "*.parquet")
    final_epoch = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max()
    common_cols = ["run_seed", "eval_verb", "eval_index_verb_context", "result_nll", "result_slor"]
    common_filter = [
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context"),
        pl.col("eval_verb").is_in(VERB_TARGET_10),
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

    arms = [
        remove_arm(scope, target)
        for scope in ("all-verb", "per-verb", "except-verb")
        for target in ("p", "e")
    ]
    return pl.concat([a.collect() for a in arms])


def run_contrast(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({metric_col: "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    ).rename({"contrast": "effect"})


def _verb_order(by: pl.Expr) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_10))
    return (
        stats.with_columns(
            (pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio")
        )
        .sort(by, descending=True)["verb"]
        .to_list()
    )


def plot_forest(contrasts: pl.DataFrame, metric_label: str, out_path: Path) -> None:
    verbs = _verb_order(pl.col("count_causative"))
    n_effects = len(CONTRASTS)
    dodge = 0.7 / n_effects
    y_pos = {v: i for i, v in enumerate(verbs)}

    fig, ax = plt.subplots(figsize=(7, max(4, len(verbs) * 0.7)))

    for idx, (effect_key, effect_label, color) in enumerate(CONTRASTS):
        sub = contrasts.filter(pl.col("effect") == effect_key)
        ordered = pl.DataFrame({"eval_verb": verbs}).join(sub, on="eval_verb", how="left")
        x = ordered["estimate"].to_numpy()
        xerr = ordered["std.error"].to_numpy() * 1.96
        y = np.array([y_pos[v] + (idx - (n_effects - 1) / 2) * dodge for v in verbs])

        ax.errorbar(
            x=x, y=y, xerr=xerr,
            fmt="o", capsize=3, color=color, markersize=5, linewidth=1.2,
            label=effect_label,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels(verbs)

    direction = "↑" if metric_label == "NLL" else "↓"
    ax.set_xlabel(
        f"DiD contrast [{metric_label}]  ({direction} = preemption-specific effect beyond entrenchment)"
    )
    ax.set_ylabel("Verb")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(title="Effect", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, metric_label: str) -> None:
    direction = "↑" if metric_label == "NLL" else "↓"
    print(f"\n{'='*64}")
    print(f"  {metric_label}  —  lexical vs abstract preemption (DiD vs e arm)")
    print("  Eval: transitive :: with_context")
    print(f"  Direction: {direction} = preemption-specific effect beyond entrenchment")
    print(f"{'='*64}")
    with pl.Config(tbl_rows=-1):
        print(
            contrasts.select("effect", "eval_verb", "estimate", "std.error", "statistic", "adj.p.value")
            .sort("effect", "eval_verb")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset-dir", type=Path, default=TESTSET_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    print("Loading testset data...")
    df = load_data()
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    for metric_key, (metric_col, metric_label) in METRICS.items():
        contrasts = run_contrast(df, metric_col)
        print_summary(contrasts, metric_label)
        out_path = args.figure_dir / f"20260513.v3.lex-vs-abstract-preemption-did_with-context_{metric_key}.pdf"
        plot_forest(contrasts, metric_label, out_path)


if __name__ == "__main__":
    main()
