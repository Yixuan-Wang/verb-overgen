"""Alternative preemption proxy: construction-preference (difference-in-differences).

Contrasts with `preemption_hypothesis_test.py` (the first-difference proxy).
Instead of contrasting `loss(transitive)` directly across `remove::p` vs
`remove::e`, this script contrasts the *gap*
    gap = loss(transitive) − loss(causative)
across the two conditions, with the gap computed per matched pair
(run_seed, eval_verb, eval_index_verb_context).

This is the proxy used by prior work. It controls for verb-level fluency by
using the grammatical counterpart as a within-condition baseline — but note
that under `remove::p :: per-verb` the causative form is itself stripped from
training, so `loss(causative | remove::p)` is mechanically inflated. The
construction-preference contrast therefore conflates the release of
preemption pressure on transitives with the mechanical loss of causative
training data. Reported here for comparability with prior work; the
first-difference proxy in the sibling script is the cleaner test.

As in `lexical_vs_abstract_preemption.py`, all three scopes (all-verb,
per-verb, except-verb) are loaded so that `abstract` and
`item_specific_given_indirect` DiD contrasts can be computed on the gap.

Expected direction of the contrast (remove::e − remove::p) on the gap:
  NLL gap   > 0  (preemption release narrows the trans−caus gap)
  SLOR gap  < 0  (preemption release narrows the (negative) trans−caus gap)
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
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

METRICS: dict[str, tuple[str, str]] = {
    "nll": ("gap_nll", "NLL Difference"),
    "slor": ("gap_slor", "SLOR Difference"),
}

CONTRASTS = [
    ("abstract", "Abstract", "#2d3748", "o"),
    ("item_specific_given_indirect", "Verb-specific", "#74add1", "s"),
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
        ), adjust = "mvt") |>
        tidy()
}
""")


def load_data(testset_dir: Path, has_context: bool) -> pl.DataFrame:
    src = pl.scan_parquet(testset_dir / "*.parquet")
    final_epoch = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max()
    join_keys = ["run_seed", "eval_verb", "eval_index_verb_context"]
    common_filter = [
        pl.col("eval_construction").is_in(["transitive", "causative"]),
        pl.col("eval_has_context") == has_context,
        pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
        final_epoch,
    ]

    def remove_arm(scope: str, target: str) -> pl.DataFrame:
        filters = [
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == scope,
            pl.col("run_target") == target,
            *common_filter,
        ]
        if scope in ("per-verb", "except-verb"):
            filters.append(pl.col("eval_verb") == pl.col("run_target_verb"))

        raw = (
            src.filter(*filters)
            .select([*join_keys, "eval_construction", "result_nll", "result_slor"])
            .collect()
        )
        assert isinstance(raw, pl.DataFrame)

        transitive = (
            raw.filter(pl.col("eval_construction") == "transitive")
            .drop("eval_construction")
            .rename({"result_nll": "nll_transitive", "result_slor": "slor_transitive"})
        )
        causative = (
            raw.filter(pl.col("eval_construction") == "causative")
            .drop("eval_construction")
            .rename({"result_nll": "nll_causative", "result_slor": "slor_causative"})
        )

        return (
            transitive.join(causative, on=join_keys, how="inner")
            .with_columns(
                (pl.col("nll_transitive") - pl.col("nll_causative")).alias("gap_nll"),
                (pl.col("slor_transitive") - pl.col("slor_causative")).alias("gap_slor"),
                pl.lit(scope).alias("scope"),
                pl.lit(target).alias("target"),
            )
            .select([*join_keys, "scope", "target", "gap_nll", "gap_slor"])
        )

    arms = [remove_arm(scope, target) for scope in ("all-verb", "per-verb", "except-verb") for target in ("p", "e")]
    return pl.concat(arms)


def run_contrast(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({metric_col: "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    ).rename({"contrast": "effect"})


def _verb_order(
    by: pl.Expr,
) -> list[str]:
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
    ax.set_yticklabels([fr"$\textit{{{verb}}}$" for verb in verbs])

    if metric_label.startswith("NLL"):
        arrow_better = r"$\uparrow^?$"
    else:
        arrow_better = r"$\downarrow^?$"
    ax.set_xlabel(
        f"{metric_label} Contrast\n" +
        rf"(\textsc{{keep-p}} $-$ \textsc{{remove-p}}) {arrow_better}"
    )
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


def print_summary(contrasts: pl.DataFrame, metric_label: str, context_tag: str) -> None:
    direction = "↑" if metric_label.startswith("NLL") else "↓"
    print(f"\n{'=' * 64}")
    print(f"  {metric_label}  —  construction-preference gap: abstract/verb-specific decomposition")
    print(f"  Eval: transitive−causative gap :: {context_tag}")
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
        default=TESTSET_DIR,
        help="Directory containing v2 testset parquets (default: %(default)s)",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Output directory for PDF figures (default: %(default)s)",
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

    print("Loading testset data...")
    df = load_data(args.testset_dir, has_context)
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} matched (transitive, causative) pairs  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    for metric_key, (metric_col, metric_label) in METRICS.items():
        contrasts = run_contrast(df, metric_col)
        print_summary(contrasts, metric_label, context_tag)

        out_path = (
            args.figure_dir
            / f"20260515.preemption-hypothesis-construction-preference_{context_tag}_{metric_key}.pdf"
        )
        plot_forest(contrasts, metric_label, out_path)


if __name__ == "__main__":
    main()
