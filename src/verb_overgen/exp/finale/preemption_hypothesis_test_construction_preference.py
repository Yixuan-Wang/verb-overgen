"""Alternative preemption proxy: construction-preference (difference-in-differences).

Contrasts with `preemption_hypothesis_test_scope_comparison.py` (the
first-difference proxy). Instead of contrasting `loss(transitive)` directly
across `remove::p` vs `remove::e`, this script contrasts the *gap*
    gap = loss(transitive) - loss(causative)
computed per matched pair (run_seed, eval_verb, eval_index_verb_context),
then takes the `remove::e - remove::p` contrast on that gap per scope.

This is the proxy used by prior work. It controls for verb-level fluency by
using the grammatical counterpart as a within-condition baseline -- but note
that under `remove::p :: per-verb` the causative form is itself stripped from
training, so `loss(causative | remove::p)` is mechanically inflated. The
construction-preference contrast therefore conflates the release of
preemption pressure on transitives with the mechanical loss of causative
training data. Reported here for comparability with prior work; the
first-difference proxy in the sibling script is the cleaner test.

Dataset, style, labels and verb order mirror
`preemption_hypothesis_test_scope_comparison.py`: all scopes are read from
`var/results/v2/testset_0818_wordlevel` plus
`var/results/v2/testset_0818_wordlevel_alternating` (which adds the per-verb
alternating eval items), concatenated. That testset is `no_context`-only and
has no SLOR, so this analysis is NLL-only and `no_context`-only.

`--tokenization bpe` instead reads `var/results/v2/testset_0818_bpe`, which
only has the `per-verb` and `all-verb` scopes, and whose `per-verb` arm only
covers the periphrastic intransitives.

Preemption hypothesis: removing the preempting causatives (p) should increase
overgeneralization more than removing entrenchment-minus-preemption (e).
Expected direction of the contrast (remove::e - remove::p) on the gap: NLL > 0.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE, VERB_ORDER
from verb_overgen.exp.finale.util import ACL_COL_WIDTH, apply_acl_style, despine
from verb_overgen.util.r_arrow_bridge import RScript

ALTERNATING_VERBS = ["stop", "break"]
VERB_TARGET = [*VERB_TARGET_FINALE, *ALTERNATING_VERBS]

TESTSET_DIRS = {
    "wordlevel": [
        Path("var/results/v2/testset_0818_wordlevel"),
        Path("var/results/v2/testset_0818_wordlevel_alternating"),
    ],
    "bpe": [Path("var/results/v2/testset_0818_bpe")],
}
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
        mutate(eval_verb = relevel(factor(eval_verb), ref = "go"))

    # Derive the cell factor and one `e - p` contrast per scope from whatever
    # scopes are actually present, ordered as they first appear in the frame.
    scopes <- unique(df$scope)
    levels <- as.vector(rbind(paste0("p_", scopes), paste0("e_", scopes)))
    df <- df |>
        mutate(cell = factor(paste(target, scope, sep = "_"), levels = levels))

    model <- lmer(
        metric ~ cell * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    method <- setNames(
        lapply(seq_along(scopes), function(i) {
            v <- rep(0, length(levels)); v[2 * i - 1] <- -1; v[2 * i] <- +1; v
        }),
        scopes
    )

    model |>
        emmeans(~ cell | eval_verb) |>
        contrast(method = method, adjust = "mvt") |>
        tidy()
}
""")


def load_data(dirs: list[Path], verbs: list[str], scopes: list[str]) -> pl.DataFrame:
    src = pl.concat(
        [pl.scan_parquet(d / "*.parquet") for d in dirs], how="vertical"
    )

    join_keys = ["run_seed", "eval_verb", "eval_index_verb_context"]
    common_filter = [
        pl.col("eval_construction").is_in(["transitive", "causative"]),
        pl.col("eval_has_context") == False,  # noqa: E712  — no_context only, always
        pl.col("eval_verb").is_in(verbs),
        # Per-run final epoch: testset dirs may have been collected with
        # different checkpoint cadences, so a global max would drop
        # shorter-logged runs.
        pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max().over("run_name"),
    ]

    def remove_arm(scope: str, target: str) -> pl.DataFrame:
        filters = [
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == scope,
            pl.col("run_target") == target,
            *common_filter,
        ]
        if scope == "per-verb":
            filters.append(pl.col("eval_verb") == pl.col("run_target_verb"))

        raw = (
            src.filter(*filters)
            .select([*join_keys, "eval_construction", "result_nll"])
            .collect()
        )

        transitive = (
            raw.filter(pl.col("eval_construction") == "transitive")
            .drop("eval_construction")
            .rename({"result_nll": "nll_transitive"})
        )
        causative = (
            raw.filter(pl.col("eval_construction") == "causative")
            .drop("eval_construction")
            .rename({"result_nll": "nll_causative"})
        )

        return (
            transitive.join(causative, on=join_keys, how="inner")
            .with_columns(
                (pl.col("nll_transitive") - pl.col("nll_causative")).alias("gap_nll"),
                pl.lit(scope).alias("scope"),
                pl.lit(target).alias("target"),
            )
            .select([*join_keys, "scope", "target", "gap_nll"])
        )

    # Empty arms (e.g. `all-intrans-only` is absent from the BPE testset)
    # contribute no rows, so the downstream contrast only sees cells that exist.
    arms = [remove_arm(scope, target) for scope in scopes for target in ("p", "e")]
    return pl.concat(arms)


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({"gap_nll": "metric"}).select(
            "scope", "target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    ).rename({"contrast": "scope"})


def _verb_order(present: set[str]) -> list[str]:
    """Follow VERB_ORDER but reversed, so the first VERB_ORDER verb sits at the
    top of the forest plot. The alternating verbs (break, stop) stay grouped at
    the bottom, separated by a visual gap. Only verbs present in the data are
    kept (the BPE testset has none of the alternating verbs)."""
    ordered = [v for v in VERB_ORDER if v in VERB_TARGET and v in present]
    periphrastic = [v for v in ordered if v not in ALTERNATING_VERBS]
    alternating = [v for v in ordered if v in ALTERNATING_VERBS]
    return list(reversed(alternating)) + list(reversed(periphrastic))


def plot_forest(contrasts: pl.DataFrame, out_path: Path, scopes=SCOPES) -> None:
    present = set(contrasts["eval_verb"].to_list())
    verbs = _verb_order(present)
    scopes = [s for s in scopes if s[0] in set(contrasts["scope"].to_list())]
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

    ax.set_xlabel(r"$\Delta$NLL Contrast (\textsc{keep-p} $-$ \textsc{remove-p}) $\uparrow$")
    ax.set_ylabel("Verb")
    ticks = ax.get_xticks()
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
    print("  NLL  —  construction-preference gap: preemption hypothesis contrast across scopes")
    print("  Eval: transitive-causative gap :: no_context")
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
        "--tokenization",
        choices=("wordlevel", "bpe"),
        default="wordlevel",
        help="Which testset to read: word-level (default) or BPE-tokenized runs.",
    )
    parser.add_argument(
        "--show-intrans-only",
        action="store_true",
        help="Include the 'Abstract (Intransitive Only)' scope in the plot (hidden by default).",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    dirs = TESTSET_DIRS[args.tokenization]
    verbs = VERB_TARGET if args.tokenization == "wordlevel" else VERB_TARGET_FINALE

    scope_keys = [s[0] for s in SCOPES]
    if not args.show_intrans_only:
        scope_keys = [s for s in scope_keys if s != "all-intrans-only"]

    print("Loading testset data...")
    df = load_data(dirs, verbs, scope_keys)
    cells = sorted({(s, t) for s, t in zip(df["scope"], df["target"], strict=True)})
    print(f"  {len(df):,} matched (transitive, causative) pairs  |  {df['eval_verb'].n_unique()} verbs  |  cells: {cells}")

    contrasts = run_contrast(df)
    print_summary(contrasts)

    scopes = [s for s in SCOPES if s[0] in scope_keys]
    suffix = "_with_intrans_only" if args.show_intrans_only else ""
    tok_tag = "" if args.tokenization == "wordlevel" else f"_{args.tokenization}"
    out_path = (
        args.figure_dir
        / f"20260830.preemption-hypothesis-construction-preference{tok_tag}_no_context_nll{suffix}.pdf"
    )
    plot_forest(contrasts, out_path, scopes=scopes)


if __name__ == "__main__":
    main()
