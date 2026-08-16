"""Decompose the verb-category preemption contrast into its two arms, anchored on `none`.

Companion to `preemption_hypothesis_test_verb_category_comparison`: that script's
count-matched contrast `remove::e - remove::p` is a difference of two removal
arms, so its sign is ambiguous (see `preemption_hypothesis_test_baseline_decomposition`
for the general argument). This script anchors both arms on `none` instead, for
each of the three verb groups (original intransitive-only, new alternating, new
transitive-only), at a chosen manipulation scope (`all-verb` or `all-intrans-only`):

- The 6 original intransitive-only verbs (`VERB_TARGET_FINALE`); `none` and the
  `all-verb` arm come from `var/results/v2/testset`, the `all-intrans-only` arm
  from `var/results/v2/testset_intransitive-only` (that directory has no `none`
  baseline of its own — `none` is scope-independent, so it is always read from
  the main `testset` directory, matching `preemption_hypothesis_test_baseline_decomposition`).
- The 7 new alternating verbs; both `none` and both scopes' `remove` arms live
  in the same directory, `var/results/v2/testset_template_alternating`.
- The 6 new transitive-only verbs; likewise all in
  `var/results/v2/testset_template_transitive-only`.

None of the three groups' own training data is touched by a *different* group's
manipulation, so any shift in a group untouched by the manipulation (e.g. the
alternating verbs under `all-intrans-only`, which only targets intransitive-only
verbs) is an indirect/cross-category effect, not a direct one.

Sign convention (transitive is the ungrammatical overgeneralization):
    arm - none < 0  ->  removal INCREASED overgeneralization
    arm - none > 0  ->  removal DECREASED overgeneralization

Preemption predicts p - none < 0. The "causatives encourage overgeneralization"
account predicts p - none > 0.

The new verbs' testsets were collected without SLOR, so this analysis is
NLL-only, and `no_context`-only (they have no `with_context` items).
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_EXTENSION, VERB_TARGET_FINALE
from verb_overgen.exp.finale.util import ACL_COL_WIDTH, apply_acl_style, despine
from verb_overgen.util.r_arrow_bridge import RScript

TESTSET_DIR = Path("var/results/v2/testset")
TESTSET_INTRANS_ONLY_DIR = Path("var/results/v2/testset_intransitive-only")
TESTSET_ALTERNATING_DIR = Path("var/results/v2/testset_template_alternating")
TESTSET_TRANSITIVE_ONLY_DIR = Path("var/results/v2/testset_template_transitive-only")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

VERB_TARGET_ALTERNATING = VERB_TARGET_EXTENSION["extended_alternating"]
VERB_TARGET_TRANSITIVE_ONLY = VERB_TARGET_EXTENSION["extended_transitive_only"]

# scope -> source dir for the original-verb group's `remove` arm (the new-verb
# groups' remove-arm dir is the same for every scope). `none` is always read
# from TESTSET_DIR for the original group, and from the group's own dir
# otherwise (see module docstring).
ORIGINAL_GROUP_REMOVE_DIR_BY_SCOPE: dict[str, Path] = {
    "all-verb": TESTSET_DIR,
    "all-intrans-only": TESTSET_INTRANS_ONLY_DIR,
}

# (group key, remove-arm dir, none-arm dir, verbs, display label)
GroupSpec = tuple[str, Path, Path, list[str], str]


def get_groups(scope: str) -> list[GroupSpec]:
    return [
        ("original", ORIGINAL_GROUP_REMOVE_DIR_BY_SCOPE[scope], TESTSET_DIR, VERB_TARGET_FINALE, "Original"),
        ("alternating", TESTSET_ALTERNATING_DIR, TESTSET_ALTERNATING_DIR, VERB_TARGET_ALTERNATING, "Alternating"),
        (
            "transitive-only",
            TESTSET_TRANSITIVE_ONLY_DIR,
            TESTSET_TRANSITIVE_ONLY_DIR,
            VERB_TARGET_TRANSITIVE_ONLY,
            "Transitive-only",
        ),
    ]


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
            target    = relevel(factor(target), ref = "none"),
            eval_verb = relevel(factor(eval_verb), ref = "go")
        )

    model <- lmer(
        metric ~ target * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    model |>
        emmeans(~ target | eval_verb) |>
        contrast(method = list(
            `p` = c(-1, +1,  0),
            `e` = c(-1,  0, +1)
        ), adjust = "mvt") |>
        tidy()
}
""")

COMMON_COLS = ["target", "run_seed", "eval_verb", "eval_index_verb_context", "result_nll"]


def load_data(scope: str, groups: list[GroupSpec]) -> pl.DataFrame:
    common_filter = [
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context") == False,  # noqa: E712
        pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
    ]

    def load_arm(source_dir: Path, verbs: list[str], target: str) -> pl.DataFrame:
        filters = [*common_filter, pl.col("eval_verb").is_in(verbs)]
        if target == "none":
            filters.append(pl.col("run_manipulation") == "none")
        else:
            filters += [
                pl.col("run_manipulation") == "remove",
                pl.col("run_scope") == scope,
                pl.col("run_target") == target,
            ]
        df = (
            pl.scan_parquet(source_dir / "*.parquet")
            .filter(*filters)
            .select(COMMON_COLS[1:])
            .with_columns(pl.lit(target).alias("target"))
            .select(COMMON_COLS)
            .collect()
        )
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0, f"No rows loaded from {source_dir} for target={target!r}, verbs={verbs}"
        return df

    arms = []
    for _, remove_dir, none_dir, verbs, _ in groups:
        arms.append(load_arm(none_dir, verbs, "none"))
        arms += [load_arm(remove_dir, verbs, target) for target, *_ in TARGETS]
    return pl.concat(arms)


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    contrasts = _R_CONTRAST.call_table(
        df.rename({"result_nll": "metric"}).select(
            "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    )
    return contrasts.rename({"contrast": "target"})


def _verb_order(verbs: list[str]) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(verbs))
    return stats.sort("count_causative", descending=True)["verb"].to_list()


def plot_decomposition(contrasts: pl.DataFrame, out_path: Path, groups: list[GroupSpec]) -> None:
    group_blocks = [(label, _verb_order(verbs)) for _, _, _, verbs, label in groups]
    verbs = [v for _, vs in group_blocks for v in vs]
    dodge = 0.7 / len(TARGETS)

    height = max(2.8, len(verbs) * 0.32 + 0.6)
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, height))

    for idx, (target, target_label, color, marker) in enumerate(TARGETS):
        sub = contrasts.filter(pl.col("target") == target)
        ordered = pl.DataFrame({"eval_verb": verbs}).join(sub, on="eval_verb", how="left")
        x = ordered["estimate"].to_numpy()
        xerr = ordered["std.error"].to_numpy() * 1.96
        y = np.arange(len(verbs)) + (idx - (len(TARGETS) - 1) / 2) * dodge

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

    y_offset = 0
    boundaries = []
    label_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for label, vs in group_blocks:
        n = len(vs)
        ax.text(
            1.02,
            y_offset + (n - 1) / 2,
            label,
            transform=label_trans,
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="center",
            rotation=-90,
        )
        y_offset += n
        boundaries.append(y_offset)

    for boundary in boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.6)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels([rf"$\textit{{{verb}}}$" for verb in verbs])
    ax.set_ylim(-0.6, len(verbs) - 0.4)
    ax.set_ylabel("Verb")

    ax.set_xlabel(r"NLL $-$ baseline \quad ($<0$: more overgen.\quad $>0$: less overgen.)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2, frameon=False)
    despine(ax)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, scope: str) -> None:
    print(f"\n{'=' * 72}")
    print("  NLL  —  baseline-anchored decomposition: original vs. alternating vs. transitive-only")
    print(f"  Eval: transitive :: no_context  |  Scope: {scope}")
    print("  Contrast: remove::<target> − none")
    print("  Sign: < 0 removal increased overgen.;  > 0 removal decreased overgen.")
    print("  Preemption predicts p − none < 0;  'causatives encourage' predicts p − none > 0.")
    print(f"{'=' * 72}")
    with pl.Config(tbl_rows=-1):
        print(
            contrasts.select("target", "eval_verb", "estimate", "std.error", "statistic", "adj.p.value").sort(
                "target", "eval_verb"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--scope",
        choices=sorted(ORIGINAL_GROUP_REMOVE_DIR_BY_SCOPE),
        default="all-verb",
        help="Manipulation scope to contrast (default: %(default)s)",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    groups = get_groups(args.scope)

    print("Loading testset data...")
    df = load_data(args.scope, groups)
    print(f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  conditions: {sorted(df['target'].unique())}")

    contrasts = run_contrast(df)
    print_summary(contrasts, args.scope)

    out_path = (
        args.figure_dir
        / f"20260713.preemption-hypothesis-verb-category-baseline-decomposition_{args.scope}_no_context_nll.pdf"
    )
    plot_decomposition(contrasts, out_path, groups)


if __name__ == "__main__":
    main()
