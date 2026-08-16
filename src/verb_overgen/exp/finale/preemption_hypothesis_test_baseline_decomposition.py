"""Decompose the preemption contrast into its two arms, anchored on the `none` baseline.

The count-matched contrast `remove::e - remove::p` (see
`preemption_hypothesis_test_scope_comparison`) is a difference of two removal
arms, so its sign is ambiguous: a negative contrast is equally consistent with
the p-arm raising NLL above baseline (causatives *encourage* overgeneralization)
and with the e-arm dropping NLL below baseline (plain entrenchment, causatives
inert). This script anchors both arms on `none` to tell those apart.

For each scope in {per-verb, all-verb, all-intrans-only} and each target in
{p, e}, contrasts that arm against `none` on `transitive` NLL at the final
training epoch, per verb.

Sign convention (transitive is the ungrammatical overgeneralization):
    arm - none < 0  ->  removal INCREASED overgeneralization
    arm - none > 0  ->  removal DECREASED overgeneralization

Preemption predicts p - none < 0. The "causatives encourage overgeneralization"
account predicts p - none > 0.

`none`, `per-verb` and `all-verb` runs come from `var/results/v2/testset`.
`all-intrans-only` runs (tag `260712-verb-category`) come from
`var/results/v2/testset_intransitive-only` and were collected without SLOR, so
this analysis is NLL-only. The two directories share an identical eval item set.
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

ACL_PAGE_WIDTH = 2 * ACL_COL_WIDTH  # two-column span, inches

# (scope, source dir, display label)
# Ordered by which causatives the p-arm removes: the target's own, everyone else's,
# everyone's, then the intransitive-only class's.
SCOPES = [
    ("per-verb", TESTSET_DIR, "Per-verb (own)"),
    ("except-verb", TESTSET_DIR, "Except-verb (others)"),
    ("all-verb", TESTSET_DIR, "All-verb"),
    ("all-intrans-only", TESTSET_INTRANS_ONLY_DIR, "All-intrans-only"),
]

# Scopes whose runs are trained per target verb, so each eval verb must be read off
# the run that targeted it.
VERB_SPECIFIC_SCOPES = frozenset({"per-verb", "except-verb"})

# (target, display label, color, marker)
TARGETS = [
    ("p", r"\textsc{remove-p} (causatives)", "#d73027", "o"),
    ("e", r"\textsc{remove-e} (matched entrench.)", "#4575b4", "s"),
]

_R_CONTRAST = RScript("""
library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

entry <- function(df) {
    df <- df |>
        as_tibble() |>
        mutate(
            cell = factor(
                paste(target, scope, sep = "_"),
                levels = c("none_none",
                           "p_per-verb",         "e_per-verb",
                           "p_except-verb",      "e_except-verb",
                           "p_all-verb",         "e_all-verb",
                           "p_all-intrans-only", "e_all-intrans-only")
            ),
            eval_verb = relevel(factor(eval_verb), ref = "go")
        )

    model <- lmer(
        metric ~ cell * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    # Cell order: 1=none, 2=p_per, 3=e_per, 4=p_except, 5=e_except,
    #             6=p_all, 7=e_all, 8=p_intrans, 9=e_intrans.
    # Every contrast is (arm - none), so the baseline carries the -1.
    model |>
        emmeans(~ cell | eval_verb) |>
        contrast(method = list(
            `p::per-verb`         = c(-1, +1,  0,  0,  0,  0,  0,  0,  0),
            `e::per-verb`         = c(-1,  0, +1,  0,  0,  0,  0,  0,  0),
            `p::except-verb`      = c(-1,  0,  0, +1,  0,  0,  0,  0,  0),
            `e::except-verb`      = c(-1,  0,  0,  0, +1,  0,  0,  0,  0),
            `p::all-verb`         = c(-1,  0,  0,  0,  0, +1,  0,  0,  0),
            `e::all-verb`         = c(-1,  0,  0,  0,  0,  0, +1,  0,  0),
            `p::all-intrans-only` = c(-1,  0,  0,  0,  0,  0,  0, +1,  0),
            `e::all-intrans-only` = c(-1,  0,  0,  0,  0,  0,  0,  0, +1)
        ), adjust = "mvt") |>
        tidy()
}
""")

# The p-arm removal sets are strictly nested for any target verb in the intransitive-only
# class: {own causatives} subset {class's causatives} subset {all causatives}, i.e.
# per-verb subset all-intrans-only subset all-verb. So removal volume is ordered by
# construction, and any account on which the effect is a monotone function of volume alone
# must order the three p-arm effects accordingly. An interior extremum — the middle scope
# differing from BOTH flanking scopes in the same direction — is impossible under any such
# account, whatever its direction. That is what this contrast family tests.
_R_MONOTONICITY = RScript("""
library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

entry <- function(df) {
    df <- df |>
        as_tibble() |>
        mutate(
            cell = factor(
                paste(target, scope, sep = "_"),
                levels = c("none_none",
                           "p_per-verb",         "e_per-verb",
                           "p_except-verb",      "e_except-verb",
                           "p_all-verb",         "e_all-verb",
                           "p_all-intrans-only", "e_all-intrans-only")
            ),
            eval_verb = relevel(factor(eval_verb), ref = "go")
        )

    model <- lmer(
        metric ~ cell * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    # Cell order: 1=none, 2=p_per, 3=e_per, 4=p_except, 5=e_except,
    #             6=p_all, 7=e_all, 8=p_intrans, 9=e_intrans.
    # Both contrasts put the middle dose (p_intrans) against a flanking dose.
    model |>
        emmeans(~ cell | eval_verb) |>
        contrast(method = list(
            `p: intrans-only - per-verb (vs smaller dose)` =
                c(0, -1, 0, 0, 0,  0, 0, +1, 0),
            `p: intrans-only - all-verb (vs larger dose)`  =
                c(0,  0, 0, 0, 0, -1, 0, +1, 0)
        ), adjust = "mvt") |>
        tidy()
}
""")

COMMON_COLS = ["run_seed", "eval_verb", "eval_index_verb_context", "result_nll"]


def load_data(has_context: bool) -> pl.DataFrame:
    common_filter = [
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context") == has_context,
        pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
        pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
    ]

    def tag(lf: pl.LazyFrame, scope: str, target: str) -> pl.LazyFrame:
        return lf.select(COMMON_COLS).with_columns(
            pl.lit(scope).alias("scope"),
            pl.lit(target).alias("target"),
        )

    def none_arm() -> pl.LazyFrame:
        # The `none` runs carry a vestigial run_target_verb ("laugh") from the metadata
        # schema; the training set is untouched, so they are not verb-specific and must
        # not be filtered on it. One run per seed, shared as the baseline for all scopes.
        src = pl.scan_parquet(TESTSET_DIR / "*.parquet")
        return tag(src.filter(pl.col("run_manipulation") == "none", *common_filter), "none", "none")

    def remove_arm(scope: str, source_dir: Path, target: str) -> pl.LazyFrame:
        src = pl.scan_parquet(source_dir / "*.parquet")
        filters = [
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == scope,
            pl.col("run_target") == target,
            *common_filter,
        ]
        if scope in VERB_SPECIFIC_SCOPES:
            filters.append(pl.col("eval_verb") == pl.col("run_target_verb"))
        return tag(src.filter(*filters), scope, target)

    arms = [none_arm()] + [
        remove_arm(scope, source_dir, target)
        for scope, source_dir, *_ in SCOPES
        for target, *_ in TARGETS
    ]
    return pl.concat([a.collect() for a in arms])


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    contrasts = _R_CONTRAST.call_table(
        df.rename({"result_nll": "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    )
    return contrasts.with_columns(
        pl.col("contrast").str.split("::").list.get(0).alias("target"),
        pl.col("contrast").str.split("::").list.get(1).alias("scope"),
    )


def run_monotonicity_test(df: pl.DataFrame) -> pl.DataFrame:
    return _R_MONOTONICITY.call_table(
        df.rename({"result_nll": "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    )


def print_monotonicity(mono: pl.DataFrame) -> None:
    print(f"\n{'=' * 72}")
    print("  Non-monotonicity test  —  is the middle dose an interior extremum?")
    print("  p-arm removal sets nest: per-verb < all-intrans-only < all-verb.")
    print("  A volume-only account requires the three effects to be ordered monotonically.")
    print("  Both contrasts significant WITH THE SAME SIGN => interior extremum => refuted.")
    print(f"{'=' * 72}")
    with pl.Config(tbl_rows=-1, fmt_str_lengths=48):
        print(
            mono.select("eval_verb", "contrast", "estimate", "std.error", "statistic", "adj.p.value").sort(
                "eval_verb", "contrast"
            )
        )


def _verb_order() -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_FINALE))
    return stats.sort("count_causative", descending=True)["verb"].to_list()


def plot_decomposition(contrasts: pl.DataFrame, out_path: Path) -> None:
    verbs = _verb_order()
    dodge = 0.7 / len(TARGETS)
    y_pos = {v: i for i, v in enumerate(verbs)}

    height = max(2.8, len(verbs) * 0.38)
    fig, axes = plt.subplots(
        1, len(SCOPES), figsize=(ACL_PAGE_WIDTH, height), sharex=True, sharey=True
    )

    for ax, (scope_key, _, scope_label) in zip(axes, SCOPES, strict=True):
        for idx, (target, target_label, color, marker) in enumerate(TARGETS):
            sub = contrasts.filter(pl.col("scope") == scope_key, pl.col("target") == target)
            ordered = pl.DataFrame({"eval_verb": verbs}).join(sub, on="eval_verb", how="left")
            x = ordered["estimate"].to_numpy()
            xerr = ordered["std.error"].to_numpy() * 1.96
            y = np.array([y_pos[v] + (idx - (len(TARGETS) - 1) / 2) * dodge for v in verbs])

            ax.errorbar(
                x=x,
                y=y,
                xerr=xerr,
                fmt=marker,
                capsize=2,
                color=color,
                markersize=3,
                linewidth=1,
                label=target_label,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(scope_label)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        despine(ax)

    axes[0].set_yticks(range(len(verbs)))
    axes[0].set_yticklabels([rf"$\textit{{{verb}}}$" for verb in verbs])
    axes[0].set_ylabel("Verb")
    axes[0].set_ylim(-0.6, len(verbs) - 0.4)

    fig.supxlabel(
        r"NLL $-$ baseline \quad ($<0$: more overgen.\quad $>0$: less overgen.)",
        fontsize=9,
        y=0.02,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, context_tag: str) -> None:
    print(f"\n{'=' * 72}")
    print("  NLL  —  baseline-anchored decomposition of the preemption contrast")
    print(f"  Eval: transitive :: {context_tag}")
    print("  Contrast: remove::<target>::<scope> − none")
    print("  Sign: < 0 removal increased overgen.;  > 0 removal decreased overgen.")
    print("  Preemption predicts p − none < 0;  'causatives encourage' predicts p − none > 0.")
    print(f"{'=' * 72}")
    with pl.Config(tbl_rows=-1):
        print(
            contrasts.select(
                "scope", "target", "eval_verb", "estimate", "std.error", "statistic", "adj.p.value"
            ).sort("scope", "target", "eval_verb")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="Use with_context eval condition instead of no_context (default: no_context)",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    has_context = args.with_context
    context_tag = "with_context" if has_context else "no_context"

    print("Loading testset data...")
    df = load_data(has_context)
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    contrasts = run_contrast(df)
    print_summary(contrasts, context_tag)

    print_monotonicity(run_monotonicity_test(df))

    out_path = args.figure_dir / f"20260713.preemption-hypothesis-baseline-decomposition_{context_tag}_nll.pdf"
    plot_decomposition(contrasts, out_path)


if __name__ == "__main__":
    main()
