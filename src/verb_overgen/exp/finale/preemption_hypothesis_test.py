"""Test the original per-verb contrast for the overall preemption effect.

Compares `remove :: p :: per-verb` vs `remove :: e :: per-verb` on
`transitive :: with_context` or `transitive :: no_context` at the final
training epoch, using a linear mixed-effects model (lme4) with per-verb
emmeans contrasts.

Preemption hypothesis: removing the preempting causatives (p) should increase
overgeneralization more than removing entrenchment-minus-preemption (e).
Expected direction of the contrast (remove::e − remove::p):
  NLL  > 0  (higher NLL = p model assigns more prob to ungrammatical transitives)
  SLOR < 0  (lower SLOR = p model considers transitives more grammatical)
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE
from verb_overgen.exp.finale.util import ACL_COL_WIDTH, apply_acl_style, despine
from verb_overgen.util.misc_util import adhoc
from verb_overgen.util.r_arrow_bridge import RScript

TESTSET_DIR = Path("var/results/v2/testset")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

METRICS: dict[str, tuple[str, str]] = {
    "nll": ("result_nll", "NLL"),
    "slor": ("result_slor", "SLOR"),
}

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


def load_data(testset_dir: Path, has_context: bool, compute_slor: bool = True) -> pl.DataFrame:
    df = (
        pl.scan_parquet(testset_dir / "*.parquet")
        .filter(
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == "per-verb",
            pl.col("run_target").is_in(["p", "e"]),
            pl.col("eval_construction") == "transitive",
            pl.col("eval_has_context") == has_context,
            pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
            pl.col("eval_verb") == pl.col("run_target_verb"),
            pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
        )
        .select(
            "run_target",
            "run_seed",
            "eval_verb",
            "eval_index_verb_context",
            "result_nll",
            *(["result_slor"] if compute_slor else []),
        )
        .collect()
    )
    assert isinstance(df, pl.DataFrame)
    return df


def run_contrast(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({metric_col: "metric"}).select(
            "run_target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    )


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

    height = max(2.2, len(verbs) * 0.28)
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, height))

    ordered_df = pl.DataFrame({"eval_verb": verbs}).join(contrasts, on="eval_verb", how="left")
    x = ordered_df["estimate"].to_numpy()
    xerr = ordered_df["std.error"].to_numpy() * 1.96

    ax.errorbar(
        x=x,
        y=list(range(len(verbs))),
        xerr=xerr,
        fmt="s",
        capsize=2,
        color="#74add1",
        markersize=3,
        linewidth=1,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels([fr"$\textit{{{verb}}}$" for verb in verbs], fontstyle="italic")

    arrow_better = r"$\uparrow$" if metric_label == "NLL" else r"$\downarrow$"
    ax.set_xlabel(
        rf"{metric_label} Contrast (\textsc{{keep-p}} $-$ \textsc{{remove-p}}) {arrow_better}"
    )
    ax.set_ylabel("Verb")
    if metric_label == "NLL":
        ticks = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10]
        ax.set_xticks(ticks)
        ax.set_xlim(ticks[0] - 0.02, ticks[-1] + 0.02)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    despine(ax)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, metric_label: str, context_tag: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {metric_label}  —  contrast: remove::e − remove::p")
    print(f"  Eval: transitive :: {context_tag}  |  Scope: per-verb")
    if metric_label == "NLL":
        print("  Hypothesis direction: contrast > 0 (preemption reduces overgen.)")
    else:
        print("  Hypothesis direction: contrast < 0 (preemption reduces overgen.)")
    print(f"{'=' * 64}")
    with pl.Config(tbl_rows=-1):
        print(contrasts.select("eval_verb", "estimate", "std.error", "statistic", "p.value").sort("eval_verb"))


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
    parser.add_argument(
        "--no-slor",
        action="store_true",
        help="Skip SLOR metric (use when testset parquets were collected without SLOR).",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    has_context = not args.no_context
    context_tag = "no_context" if args.no_context else "with_context"
    compute_slor = not args.no_slor

    print("Loading testset data...")
    df = load_data(args.testset_dir, has_context, compute_slor)
    n_verbs = df["eval_verb"].n_unique()
    n_rows = len(df)
    print(f"  {n_rows:,} rows  |  {n_verbs} verbs  |  conditions: {sorted(df['run_target'].unique())}")

    metrics = {k: v for k, v in METRICS.items() if k != "slor" or compute_slor}
    for metric_key, (metric_col, metric_label) in metrics.items():
        contrasts = run_contrast(df, metric_col)
        print_summary(contrasts, metric_label, context_tag)

        out_path = args.figure_dir / f"20260511.preemption-hypothesis_per-verb_{context_tag}_{metric_key}.pdf"
        plot_forest(contrasts, metric_label, out_path)


if __name__ == "__main__":
    main()
