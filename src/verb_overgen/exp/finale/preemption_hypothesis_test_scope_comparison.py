"""Compare the preemption hypothesis contrast across three manipulation scopes.

For each scope in {per-verb, all-verb, all-intrans-only}, contrasts
`remove::e::scope` against `remove::p::scope` (both count-matched) on
`transitive :: with_context` or `transitive :: no_context` NLL at the final
training epoch, per verb, and plots the three scopes side by side.

`per-verb` and `all-verb` runs come from `var/results/v2/testset`.
`all-intrans-only` runs (tag `260712-verb-category`) come from
`var/results/v2/testset_intransitive-only` and were collected without SLOR,
so this analysis is NLL-only.

Preemption hypothesis: removing the preempting causatives (p) should increase
overgeneralization more than removing entrenchment-minus-preemption (e).
Expected direction of the contrast (remove::e − remove::p): NLL > 0.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE
from verb_overgen.exp.finale.util import ACL_COL_WIDTH, apply_acl_style, despine
from verb_overgen.util.r_arrow_bridge import RScript

TESTSET_DIR = Path("var/results/v2/testset")
TESTSET_INTRANS_ONLY_DIR = Path("var/results/v2/testset_intransitive-only")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

# (scope, source dir, display label, color, marker)
SCOPES = [
    ("per-verb", TESTSET_DIR, "Per-verb", "#74add1", "s"),
    ("all-verb", TESTSET_DIR, "All-verb", "#2d3748", "o"),
    ("all-intrans-only", TESTSET_INTRANS_ONLY_DIR, "All-intrans-only", "#f46d43", "^"),
]

_R_CONTRAST = RScript("""
library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

entry <- function(df) {
    df <- df |>
        as_tibble() |>
        mutate(
            cell = factor(
                paste(target, scope, sep = "_"),
                levels = c("p_per-verb",         "e_per-verb",
                           "p_all-verb",         "e_all-verb",
                           "p_all-intrans-only", "e_all-intrans-only")
            ),
            eval_verb = relevel(factor(eval_verb), ref = "go")
        )

    model <- lmer(
        metric ~ cell * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    # Cell order: 1=p_per, 2=e_per, 3=p_all, 4=e_all, 5=p_intrans, 6=e_intrans.
    model |>
        emmeans(~ cell | eval_verb) |>
        contrast(method = list(
            `per-verb`         = c(-1, +1,  0,  0,  0,  0),
            `all-verb`         = c( 0,  0, -1, +1,  0,  0),
            `all-intrans-only` = c( 0,  0,  0,  0, -1, +1)
        ), adjust = "mvt") |>
        tidy()
}
""")


def load_data(has_context: bool) -> pl.DataFrame:
    common_cols = ["run_seed", "eval_verb", "eval_index_verb_context", "result_nll"]
    common_filter = [
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context") == has_context,
        pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
        pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
    ]

    def remove_arm(scope: str, source_dir: Path, target: str) -> pl.LazyFrame:
        src = pl.scan_parquet(source_dir / "*.parquet")
        filters = [
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == scope,
            pl.col("run_target") == target,
            *common_filter,
        ]
        if scope == "per-verb":
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
        remove_arm(scope, source_dir, target)
        for scope, source_dir, *_ in SCOPES
        for target in ("p", "e")
    ]
    return pl.concat([a.collect() for a in arms])


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({"result_nll": "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    ).rename({"contrast": "scope"})


def _verb_order(by: pl.Expr) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_FINALE))
    return (
        stats.with_columns((pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio"))
        .sort(by, descending=True)["verb"]
        .to_list()
    )


def plot_forest(contrasts: pl.DataFrame, out_path: Path) -> None:
    verbs = _verb_order(pl.col("count_causative"))
    n_scopes = len(SCOPES)
    dodge = 0.7 / n_scopes
    y_pos = {v: i for i, v in enumerate(verbs)}

    height = max(2.8, len(verbs) * 0.35)
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, height))

    for idx, (scope_key, _, scope_label, color, marker) in enumerate(SCOPES):
        sub = contrasts.filter(pl.col("scope") == scope_key)
        ordered = pl.DataFrame({"eval_verb": verbs}).join(sub, on="eval_verb", how="left")
        x = ordered["estimate"].to_numpy()
        xerr = ordered["std.error"].to_numpy() * 1.96
        y = np.array([y_pos[v] + (idx - (n_scopes - 1) / 2) * dodge for v in verbs])

        ax.errorbar(
            x=x,
            y=y,
            xerr=xerr,
            fmt=marker,
            capsize=2,
            color=color,
            markersize=3,
            linewidth=1,
            label=scope_label,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels([fr"$\textit{{{verb}}}$" for verb in verbs])

    ax.set_xlabel(r"NLL Contrast (\textsc{keep-p} $-$ \textsc{remove-p}) $\uparrow$")
    ax.set_ylabel("Verb")
    ticks = ax.get_xticks()
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


def print_summary(contrasts: pl.DataFrame, context_tag: str) -> None:
    print(f"\n{'=' * 64}")
    print("  NLL  —  preemption hypothesis contrast across scopes")
    print(f"  Eval: transitive :: {context_tag}")
    print("  Contrast: remove::e − remove::p (per scope)")
    print("  Hypothesis direction: contrast > 0 (preemption reduces overgen.)")
    print(f"{'=' * 64}")
    with pl.Config(tbl_rows=-1):
        print(
            contrasts.select("scope", "eval_verb", "estimate", "std.error", "statistic", "adj.p.value").sort(
                "scope", "eval_verb"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    df = load_data(has_context)
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    contrasts = run_contrast(df)
    print_summary(contrasts, context_tag)

    out_path = args.figure_dir / f"20260713.preemption-hypothesis-scope-comparison_{context_tag}_nll.pdf"
    plot_forest(contrasts, out_path)


if __name__ == "__main__":
    main()
