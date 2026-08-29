"""Compare the preemption contrast across original, alternating, and transitive-only verbs.

Contrasts `remove::e::<scope>` against `remove::p::<scope>` on
`transitive :: no_context` NLL at the final training epoch, per eval verb,
for three verb groups, at a chosen manipulation scope (`all-verb` or
`all-intrans-only`):

- The original intransitive-only verbs (`VERB_TARGET_FINALE`).
- The alternating verbs (`VERB_TARGET_EXTENSION["extended_alternating"]`).
- The transitive-only verbs (`VERB_TARGET_EXTENSION["extended_transitive_only"]`).

All groups and both scopes are read from the single
`var/results/v2/testset_0818_wordlevel` directory, which bundles every
`run_scope` (distinguished by the `run_scope` column) for every eval verb.
Only the new-group verbs that actually have testset runs there are shown
(currently `break`, `close`, `open`, `roll` for alternating; `touch`, `hit`,
`feed` for transitive-only).

All three groups share the same scope's training runs (the manipulation
removes causatives corpus-wide for `all-verb`, or across the intransitive-only
verb category for `all-intrans-only`; either way it also touches the new
verbs' causative instances even though they were never a manipulation target
themselves). There is no per-verb removal training for the new verbs, so these
coarser scopes are the only contrasts available for them; this script uses
them for all three groups to keep the comparison apples-to-apples. The new
verbs' testsets were collected without SLOR, so this analysis is NLL-only.

Only `no_context` is used: the new verbs' testsets have no `with_context` items
(their `kind` values are just `sentence_causative`/`sentence_transitive`).

Preemption hypothesis: removing the preempting causatives (p) should increase
overgeneralization more than removing entrenchment-minus-preemption (e).
Expected direction of the contrast (remove::e − remove::p): NLL > 0.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from verb_overgen.common.features import VERB_TARGET_EXTENSION, VERB_TARGET_FINALE
from verb_overgen.exp.finale.util import ACL_COL_WIDTH, apply_acl_style, despine
from verb_overgen.util.r_arrow_bridge import RScript

# All groups and scopes now live in this one directory, distinguished by the
# `run_scope` column rather than by separate directories.
TESTSET_DIR = Path("var/results/v2/testset_0818_wordlevel")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

VERB_TARGET_ALTERNATING = VERB_TARGET_EXTENSION["extended_alternating"]
VERB_TARGET_TRANSITIVE_ONLY = VERB_TARGET_EXTENSION["extended_transitive_only"]

SCOPES = ["all-verb", "all-intrans-only"]


def get_groups() -> list[tuple[str, list[str], str, str]]:
    # (group key, verbs, display label, color)
    return [
        ("original", VERB_TARGET_FINALE, "Original", "#74add1"),
        ("alternating", VERB_TARGET_ALTERNATING, "Alternating", "#f46d43"),
        ("transitive-only", VERB_TARGET_TRANSITIVE_ONLY, "Transitive-only", "#009e73"),
    ]


_R_CONTRAST = RScript("""
library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

entry <- function(df) {
    df <- df |>
        as_tibble() |>
        mutate(
            run_target = relevel(factor(run_target), ref = "e"),
            eval_verb  = relevel(factor(eval_verb),  ref = "go")
        )

    model <- lmer(
        metric ~ run_target * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    model |>
        emmeans(pairwise ~ run_target | eval_verb) |>
        (\(x) x$contrasts)() |>
        tidy()
}
""")

COMMON_COLS = ["run_target", "run_seed", "eval_verb", "eval_index_verb_context", "result_nll"]


def load_data(scope: str, groups: list[tuple[str, list[str], str, str]]) -> pl.DataFrame:
    def load_group(verbs: list[str]) -> pl.DataFrame:
        df = (
            pl.scan_parquet(TESTSET_DIR / "*.parquet")
            .filter(
                pl.col("run_manipulation") == "remove",
                pl.col("run_scope") == scope,
                pl.col("run_target").is_in(["p", "e"]),
                pl.col("eval_construction") == "transitive",
                pl.col("eval_has_context") == False,  # noqa: E712
                pl.col("eval_verb").is_in(verbs),
                pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
            )
            .select(COMMON_COLS)
            .collect()
        )
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0, f"No rows loaded from {TESTSET_DIR} for scope {scope}, verbs {verbs}"
        return df

    return pl.concat([load_group(verbs) for _, verbs, *_ in groups])


def run_contrast(df: pl.DataFrame) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({"result_nll": "metric"}).select(
            "run_target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    )


def _verb_order(verbs: list[str]) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(verbs))
    return (
        stats.with_columns((pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio"))
        .sort("count_causative", descending=True)["verb"]
        .to_list()
    )


def plot_forest(contrasts: pl.DataFrame, out_path: Path, groups: list[tuple[str, list[str], str, str]]) -> None:
    present = set(contrasts["eval_verb"].to_list())
    group_verbs = [(label, color, [v for v in _verb_order(verbs) if v in present]) for _, verbs, label, color in groups]
    verbs = [v for _, _, vs in group_verbs for v in vs]

    height = max(2.6, len(verbs) * 0.3 + 0.6)
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, height))

    ordered_df = pl.DataFrame({"eval_verb": verbs}).join(contrasts, on="eval_verb", how="left")
    x = ordered_df["estimate"].to_numpy()
    xerr = ordered_df["std.error"].to_numpy() * 1.96

    y_offset = 0
    boundaries = []
    for label, color, vs in group_verbs:
        n = len(vs)
        y = range(y_offset, y_offset + n)
        ax.errorbar(
            x=x[y_offset : y_offset + n],
            y=list(y),
            xerr=xerr[y_offset : y_offset + n],
            fmt="s",
            capsize=2,
            color=color,
            markersize=3,
            linewidth=1,
            label=label,
        )
        y_offset += n
        boundaries.append(y_offset)

    for boundary in boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.6)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels([rf"$\textit{{{verb}}}$" for verb in verbs])
    ax.set_ylim(-0.6, len(verbs) - 0.4)

    ax.set_xlabel(r"NLL Contrast (\textsc{keep-p} $-$ \textsc{remove-p}) $\uparrow$")
    ax.set_ylabel("Verb")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=len(group_verbs), frameon=False)
    despine(ax)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, scope: str) -> None:
    print(f"\n{'=' * 64}")
    print("  NLL  —  preemption contrast: original vs. alternating vs. transitive-only verbs")
    print(f"  Eval: transitive :: no_context  |  Scope: {scope}")
    print("  Contrast: remove::e − remove::p")
    print("  Hypothesis direction: contrast > 0 (preemption reduces overgen.)")
    print(f"{'=' * 64}")
    with pl.Config(tbl_rows=-1):
        print(contrasts.select("eval_verb", "estimate", "std.error", "statistic", "p.value").sort("eval_verb"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="all-verb",
        help="Manipulation scope to contrast (default: %(default)s)",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    groups = get_groups()

    print("Loading testset data...")
    df = load_data(args.scope, groups)
    print(
        f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  conditions: {sorted(df['run_target'].unique())}"
    )

    contrasts = run_contrast(df)
    print_summary(contrasts, args.scope)

    out_path = (
        args.figure_dir / f"20260829.preemption-hypothesis-verb-category-comparison_{args.scope}_no_context_nll.pdf"
    )
    plot_forest(contrasts, out_path, groups)


if __name__ == "__main__":
    main()
