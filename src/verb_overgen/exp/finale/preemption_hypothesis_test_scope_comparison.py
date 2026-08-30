"""Compare the preemption hypothesis contrast across three manipulation scopes.

For each scope in {per-verb, all-verb, all-intrans-only}, contrasts
`remove::e::scope` against `remove::p::scope` (both count-matched) on
`transitive :: no_context` NLL at the final training epoch, per verb, and
plots the three scopes side by side.

All three scopes are read from `var/results/v2/testset_0818_wordlevel` plus
`var/results/v2/testset_0818_wordlevel_alternating` (which adds the per-verb
alternating eval items), concatenated. Each bundles every `run_scope`
(distinguished by the `run_scope` column) for every eval verb. That testset
has no `with_context` items and no SLOR, so this analysis is NLL-only and
`no_context`-only.

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

ALTERNATING_VERBS = ["stop", "break"]
VERB_TARGET = [*VERB_TARGET_FINALE, *ALTERNATING_VERBS]

# All scopes live in these directories, distinguished by the `run_scope`
# column rather than by separate directories. The `_alternating` dir carries
# the per-verb alternating eval items and is concatenated in.
TESTSET_DIRS = [
    Path("var/results/v2/testset_0818_wordlevel"),
    Path("var/results/v2/testset_0818_wordlevel_alternating"),
]
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

# (scope, display label, color, marker)
SCOPES = [
    ("per-verb", "Verb-specific", "#74add1", "s"),
    ("all-verb", "Abstract", "#2d3748", "o"),
    ("all-intrans-only", "Abstract (Intransitive Only)", "#4c6a8f", "^"),
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


def load_data() -> pl.DataFrame:
    src = pl.concat(
        [pl.scan_parquet(d / "*.parquet") for d in TESTSET_DIRS], how="vertical"
    )

    common_cols = ["run_seed", "eval_verb", "eval_index_verb_context", "result_nll"]
    common_filter = [
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context") == False,  # noqa: E712
        pl.col("eval_verb").is_in(VERB_TARGET),
        # Per-run final epoch: testset dirs may have been collected with
        # different checkpoint cadences, so a global max would drop
        # shorter-logged runs.
        pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max().over("run_name"),
    ]

    def remove_arm(scope: str, target: str) -> pl.LazyFrame:
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

    arms = [remove_arm(scope, target) for scope, *_ in SCOPES for target in ("p", "e")]
    return pl.concat([a.collect() for a in arms])


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({"result_nll": "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    ).rename({"contrast": "scope"})


def _verb_order(by: pl.Expr) -> list[str]:
    """Sort the periphrastic and alternating verb groups independently, then
    stack them: alternating (break, stop) first, periphrastic last."""
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET))
    ranked = stats.with_columns(
        (pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio")
    ).sort(by, descending=True)["verb"].to_list()
    periphrastic = [v for v in ranked if v not in ALTERNATING_VERBS]
    alternating = [v for v in ranked if v in ALTERNATING_VERBS]
    return alternating + periphrastic


def plot_forest(contrasts: pl.DataFrame, out_path: Path, scopes=SCOPES) -> None:
    verbs = _verb_order(pl.col("count_causative"))
    n_scopes = len(scopes)
    dodge = 0.7 / n_scopes
    n_alt = sum(v in ALTERNATING_VERBS for v in verbs)
    gap = 1.0
    y_pos = {
        v: i if i < n_alt else i + gap
        for i, v in enumerate(verbs)
    }

    height = max(2.8, len(verbs) * 0.35)
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, height))

    for idx, (scope_key, scope_label, color, marker) in enumerate(scopes):
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

    y_ticks = [y_pos[v] for v in verbs]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([rf"$\textit{{{verb}}}$" for verb in verbs])

    ax.set_xlabel(r"NLL Contrast (\textsc{keep-p} $-$ \textsc{remove-p}) $\uparrow$")
    ax.set_ylabel("Verb")
    ticks = np.linspace(-0.18, 0.18, num=7)
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0] - 0.02, ticks[-1] + 0.02)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=n_scopes, frameon=False)
    despine(ax)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame) -> None:
    print(f"\n{'=' * 64}")
    print("  NLL  —  preemption hypothesis contrast across scopes")
    print("  Eval: transitive :: no_context")
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
        "--show-intrans-only",
        action="store_true",
        help="Include the 'Abstract (Intransitive Only)' scope in the plot (hidden by default).",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    print("Loading testset data...")
    df = load_data()
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    contrasts = run_contrast(df)
    print_summary(contrasts)

    scopes = SCOPES if args.show_intrans_only else [
        s for s in SCOPES if s[0] != "all-intrans-only"
    ]
    suffix = "_with_intrans_only" if args.show_intrans_only else ""
    out_path = (
        args.figure_dir
        / f"20260830.preemption-hypothesis-scope-comparison_no_context_nll{suffix}.pdf"
    )
    plot_forest(contrasts, out_path, scopes=scopes)


if __name__ == "__main__":
    main()
