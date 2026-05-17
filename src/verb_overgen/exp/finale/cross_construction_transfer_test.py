"""Cross-construction transfer: per-verb mixed-model contrasts.

Two *separate* linear mixed-effects models (lme4), each at the final
training epoch on `with_context` or `no_context` evals, with per-verb
`revpairwise` emmeans contrasts:

  transitive : remove::e::per-verb  vs  remove::p::per-verb
      Does the presence of periphrastic causatives feed transitive
      overgeneralization?  (causative -> transitive)
  causative  : substitute::p::per-verb  vs  remove::p::per-verb
      Does the presence of (ungrammatical) transitives feed causative
      knowledge?  (transitive -> causative)

The models are kept separate on purpose: the condition sets only partially
overlap, the two evals are different test instances, and transitive vs
causative metrics are not on a commensurable scale. Compare the two
contrasts qualitatively (direction + significance), not by raw magnitude.

Both contrasts are framed "construction present - construction absent":
  transitive:  e - p                 (e keeps causatives, p removes them)
  causative:   substitute - remove    (substitute injects transitives,
                                       remove leaves the slots empty)

Direction (sign convention shared with lexical_vs_abstract_preemption.py):
  NLL > 0 / SLOR < 0  ->  competition / preemption-style effect
  NLL < 0 / SLOR > 0  ->  positive cross-construction generalization
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    eval_construction: str
    condition_col: str  # column whose values distinguish the two arms
    condition_filter: pl.Expr  # fixes the rest of the run condition
    condition_values: list[str]  # the two arms (present, absent order is free)
    ref_level: str  # revpairwise -> (non-ref - ref); pick `absent` arm as ref
    contrast_label: str  # human-readable "<present> - <absent>"


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="transitive",
        eval_construction="transitive",
        condition_col="run_target",
        condition_filter=pl.col("run_manipulation") == "remove",
        condition_values=["e", "p"],
        ref_level="p",  # -> e - p
        contrast_label="remove::e - remove::p",
    ),
    ModelSpec(
        name="causative",
        eval_construction="causative",
        condition_col="run_manipulation",
        condition_filter=pl.col("run_target") == "p",
        condition_values=["remove", "substitute"],
        ref_level="remove",  # -> substitute - remove
        contrast_label="substitute::p - remove::p",
    ),
]

_R_CONTRAST = RScript("""
library(arrow); library(dplyr); library(lme4); library(emmeans); library(broom)

entry <- function(df) {
    ref <- as.character(df$ref_level[1])
    df <- df |>
        as_tibble() |>
        mutate(
            condition = relevel(factor(condition), ref = ref),
            eval_verb = relevel(factor(eval_verb), ref = "go")
        )

    model <- lmer(
        metric ~ condition * eval_verb + (1 | run_seed:eval_verb:eval_index_verb_context),
        data = df
    )

    model |>
        emmeans(revpairwise ~ condition | eval_verb) |>
        (\(x) x$contrasts)() |>
        tidy()
}
""")


def load_data(testset_dir: Path, spec: ModelSpec, has_context: bool) -> pl.DataFrame:
    df = (
        pl.scan_parquet(testset_dir / "*.parquet")
        .filter(
            pl.col("run_scope") == "per-verb",
            spec.condition_filter,
            pl.col(spec.condition_col).is_in(spec.condition_values),
            pl.col("eval_construction") == spec.eval_construction,
            pl.col("eval_has_context") == has_context,
            pl.col("eval_verb").is_in(VERB_TARGET_10),
            pl.col("eval_verb") == pl.col("run_target_verb"),
            pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
        )
        .select(
            pl.col(spec.condition_col).alias("condition"),
            "run_seed",
            "eval_verb",
            "eval_index_verb_context",
            "result_nll",
            "result_slor",
        )
        .collect()
    )
    assert isinstance(df, pl.DataFrame)
    return df


def run_contrast(df: pl.DataFrame, metric_col: str, ref_level: str) -> pl.DataFrame:
    return _R_CONTRAST.call_table(
        df.rename({metric_col: "metric"})
        .select("condition", "run_seed", "eval_verb", "eval_index_verb_context", "metric")
        .with_columns(pl.lit(ref_level).alias("ref_level"))
    )


def _verb_order() -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_10))
    return stats.sort("count_causative", descending=True)["verb"].to_list()


def plot_forest(
    contrasts: pl.DataFrame,
    spec: ModelSpec,
    metric_label: str,
    out_path: Path,
    context_tag: str = "with_context",
) -> None:
    verbs = _verb_order()

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

    # NLL < 0 / SLOR > 0 = positive cross-construction generalization.
    transfer_side = "←" if metric_label == "NLL" else "→"
    ax.set_xlabel(
        f"Contrast ({spec.contrast_label}) [{metric_label}]"
        f"   ({transfer_side} = positive cross-construction generalization)"
    )
    ax.set_ylabel("Verb")
    ax.set_title(f"{spec.eval_construction} :: {context_tag}")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_summary(
    contrasts: pl.DataFrame,
    spec: ModelSpec,
    metric_label: str,
    context_tag: str = "with_context",
) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {metric_label}  —  contrast: {spec.contrast_label}")
    print(f"  Eval: {spec.eval_construction} :: {context_tag}  |  Scope: per-verb")
    if metric_label == "NLL":
        print("  contrast < 0: positive cross-construction generalization")
        print("  contrast > 0: competition / preemption-style effect")
    else:
        print("  contrast > 0: positive cross-construction generalization")
        print("  contrast < 0: competition / preemption-style effect")
    print(f"{'=' * 64}")
    with pl.Config(tbl_rows=-1):
        print(contrasts.select("eval_verb", "estimate", "std.error", "statistic", "p.value").sort("eval_verb"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset-dir", type=Path, default=TESTSET_DIR)
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

    for spec in MODELS:
        print(f"\nLoading testset data for `{spec.name}` model...")
        df = load_data(args.testset_dir, spec, has_context)
        n_verbs = df["eval_verb"].n_unique()
        print(f"  {len(df):,} rows  |  {n_verbs} verbs  |  conditions: {sorted(df['condition'].unique())}")

        for metric_key, (metric_col, metric_label) in METRICS.items():
            contrasts = run_contrast(df, metric_col, spec.ref_level)
            print_summary(contrasts, spec, metric_label, context_tag)

            out_path = args.figure_dir / (
                f"20260514.cross-construction-transfer_{spec.name}_{context_tag}_{metric_key}.pdf"
            )
            plot_forest(contrasts, spec, metric_label, out_path, context_tag)


if __name__ == "__main__":
    main()
