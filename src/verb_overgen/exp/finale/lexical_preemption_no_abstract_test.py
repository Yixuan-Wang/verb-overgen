"""Isolate lexical preemption: `e :: per-verb-no-abstract` vs `p :: all-verb`.

Both conditions remove every other verb's causatives (abstract preemption off in
both arms) and reduce the target verb's total count by the same number.  The only
difference: `p :: all-verb` removes the target verb's *own* causatives (lexical
preemption removed), while `e :: per-verb-no-abstract` keeps them and instead
removes an equivalent count from the target's entrenchment.

Contrast direction: e::per-verb-no-abstract − p::all-verb.
  NLL  > 0  (e has higher NLL → p model assigns more prob to transitives)
  SLOR < 0  (e has lower SLOR → p model considers transitives more grammatical)
indicates lexical preemption is operative.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE
from verb_overgen.util.r_arrow_bridge import RScript

TESTSET_DIR = Path("var/results/v2/testset")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

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
            cell      = relevel(factor(cell), ref = "p_all"),
            eval_verb = relevel(factor(eval_verb), ref = "go")
        )

    model <- lmer(
        metric ~ cell * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    # e_noabs − p_all: positive = lexical preemption operative
    model |>
        emmeans(~ cell | eval_verb) |>
        contrast(method = list(lex_preemption = c(-1, +1)), adjust = "mvt") |>
        tidy()
}
""")


def load_data(testset_dir: Path, has_context: bool) -> pl.DataFrame:
    src = pl.scan_parquet(testset_dir / "*.parquet")
    common_cols = ["run_seed", "eval_verb", "eval_index_verb_context", "result_nll", "result_slor"]
    common_filters = [
        pl.col("run_manipulation") == "remove",
        pl.col("eval_construction") == "transitive",
        pl.col("eval_has_context") == has_context,
        pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
        pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
    ]

    p_all = (
        src.filter(
            *common_filters,
            pl.col("run_scope") == "all-verb",
            pl.col("run_target") == "p",
        )
        .select(common_cols)
        .with_columns(pl.lit("p_all").alias("cell"))
    )

    e_noabs = (
        src.filter(
            *common_filters,
            pl.col("run_scope") == "per-verb-no-abstract",
            pl.col("run_target") == "e",
            pl.col("eval_verb") == pl.col("run_target_verb"),
        )
        .select(common_cols)
        .with_columns(pl.lit("e_noabs").alias("cell"))
    )

    return pl.concat([p_all.collect(), e_noabs.collect()])


def run_contrast(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({metric_col: "metric"}).select("cell", "run_seed", "eval_verb", "eval_index_verb_context", "metric")
    )


def _verb_order(by: pl.Expr) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_FINALE))
    return (
        stats.with_columns((pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio"))
        .sort(by, descending=True)["verb"]
        .to_list()
    )


def plot_forest(contrasts: pl.DataFrame, metric_label: str, out_path: Path) -> None:
    verbs = _verb_order(pl.col("count_causative"))

    fig, ax = plt.subplots(figsize=(7, max(4, len(verbs) * 0.55)))

    ordered_df = pl.DataFrame({"eval_verb": verbs}).join(contrasts, on="eval_verb", how="left")
    x = ordered_df["estimate"].to_numpy()
    xerr = ordered_df["std.error"].to_numpy() * 1.96

    ax.errorbar(
        x=x,
        y=list(range(len(verbs))),
        xerr=xerr,
        fmt="o",
        capsize=4,
        color="black",
        markersize=6,
        linewidth=1.5,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(verbs)))
    ax.set_yticklabels(verbs)

    if metric_label == "NLL":
        arrow_better = "↑"
    else:
        arrow_better = "↓"
    ax.set_xlabel(f"Contrast (e::no-abstract − p::all-verb){arrow_better} [{metric_label}]")
    ax.set_ylabel("Verb")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(contrasts: pl.DataFrame, metric_label: str, context_tag: str) -> None:
    if metric_label == "NLL":
        direction = "contrast > 0 (lexical preemption reduces overgen.)"
    else:
        direction = "contrast < 0 (lexical preemption reduces overgen.)"
    print(f"\n{'=' * 64}")
    print(f"  {metric_label}  —  contrast: e::per-verb-no-abstract − p::all-verb")
    print(f"  Eval: transitive :: {context_tag}")
    print(f"  Hypothesis direction: {direction}")
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
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    has_context = not args.no_context
    context_tag = "no_context" if args.no_context else "with_context"

    print("Loading testset data...")
    df = load_data(args.testset_dir, has_context)
    cells = sorted(df["cell"].unique().to_list())
    n_verbs = df["eval_verb"].n_unique()
    print(f"  {len(df):,} rows  |  {n_verbs} verbs  |  cells: {cells}")

    for metric_key, (metric_col, metric_label) in METRICS.items():
        contrasts = run_contrast(df, metric_col)
        print_summary(contrasts, metric_label, context_tag)

        out_path = args.figure_dir / f"20260519.lexical-preemption-no-abstract_{context_tag}_{metric_key}.pdf"
        plot_forest(contrasts, metric_label, out_path)


if __name__ == "__main__":
    main()
