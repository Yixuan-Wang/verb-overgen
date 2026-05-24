"""Dumbbell plot for three conditions.

Compares `none`, `remove :: e :: per-verb`, and `remove :: p :: per-verb` at
the final training epoch, per verb.  Each dumbbell connects the causative end
(circle) to the transitive end (triangle) for the same condition.

Colors: none=black, remove::e=blue, remove::p=red.
"""

import argparse
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from verb_overgen.common.features import VERB_TARGET_FINALE
from verb_overgen.exp.finale.util import ACL_COL_WIDTH, apply_acl_style, despine
from verb_overgen.util.misc_util import adhoc

TESTSET_DIR = Path("var/results/v2/testset")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")
FIGURE_DIR = Path("var/figures")

apply_acl_style()

METRICS: dict[str, tuple[str, str]] = {
    "nll": ("result_nll", "NLL"),
    "slor": ("result_slor", "SLOR"),
}

CONDITIONS = [
    # ("none", "black"),
    ("remove::e", r"\textsc{keep-p}", "tab:orange"),
    ("remove::p", r"\textsc{remove-p}", "tab:blue"),
]

MARKERS = {"causative": 
    ("expected", "o"),
    "transitive": ("overgeneralized", "X"),
}


def _verb_order(by: pl.Expr) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(VERB_TARGET_FINALE))
    return (
        stats.with_columns((pl.col("count_causative") / pl.col("count_verb_like")).alias("causative_ratio"))
        .sort(by, descending=True)["verb"]
        .to_list()
    )


def load_data(testset_dir: Path, has_context: bool) -> pl.DataFrame:
    src = pl.scan_parquet(testset_dir / "*.parquet")
    final_epoch = pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max()

    # none = (
    #     src.filter(
    #         pl.col("run_manipulation") == "none",
    #         pl.col("eval_construction").is_in(["causative", "transitive"]),
    #         pl.col("eval_has_context") == has_context,
    #         pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
    #         final_epoch,
    #     )
    #     .select("run_seed", "eval_verb", "eval_construction", "eval_index_verb_context", "result_nll", "result_slor")
    #     .with_columns(pl.lit("none").alias("condition"))
    # )

    remove_p = (
        src.filter(
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == "per-verb",
            pl.col("run_target") == "p",
            pl.col("eval_construction").is_in(["causative", "transitive"]),
            pl.col("eval_has_context") == has_context,
            pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
            pl.col("eval_verb") == pl.col("run_target_verb"),
            final_epoch,
        )
        .select("run_seed", "eval_verb", "eval_construction", "eval_index_verb_context", "result_nll", "result_slor")
        .with_columns(pl.lit("remove::p").alias("condition"))
    )

    remove_e = (
        src.filter(
            pl.col("run_manipulation") == "remove",
            pl.col("run_scope") == "per-verb",
            pl.col("run_target") == "e",
            pl.col("eval_construction").is_in(["causative", "transitive"]),
            pl.col("eval_has_context") == has_context,
            pl.col("eval_verb").is_in(VERB_TARGET_FINALE),
            pl.col("eval_verb") == pl.col("run_target_verb"),
            final_epoch,
        )
        .select("run_seed", "eval_verb", "eval_construction", "eval_index_verb_context", "result_nll", "result_slor")
        .with_columns(pl.lit("remove::e").alias("condition"))
    )

    return pl.concat(
        [
            # none.collect(),
            remove_p.collect(),
            remove_e.collect(),
        ]
    )


def aggregate(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    return (
        df.group_by("condition", "eval_verb", "eval_construction")
        .agg(pl.col(metric_col).mean().alias("metric_mean"))
        .sort("eval_verb", "condition", "eval_construction")
    )


def plot_dumbbell(agg: pl.DataFrame, metric_label: str, context_label: str, out_path: Path) -> None:
    eval_verbs = _verb_order(pl.col("count_causative"))
    x_pos = {v: i for i, v in enumerate(eval_verbs)}
    n = len(CONDITIONS)
    dodge = 0.8 / n

    df = agg.to_pandas()
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, 3.0))

    for idx, (cond, label, color) in enumerate(CONDITIONS):
        cond_df = df[df["condition"] == cond]
        x_offset = (idx - (n - 1) / 2) * dodge

        for verb in eval_verbs:
            caus = cond_df[(cond_df["eval_construction"] == "causative") & (cond_df["eval_verb"] == verb)]
            tran = cond_df[(cond_df["eval_construction"] == "transitive") & (cond_df["eval_verb"] == verb)]
            if caus.empty or tran.empty:
                continue
            xv = x_pos[verb] + x_offset
            ax.plot(
                [xv, xv],
                [caus["metric_mean"].values[0], tran["metric_mean"].values[0]],
                color=color,
                linewidth=1,
                alpha=0.6,
                zorder=1,
                label=label,
            )

        for construction, (label, marker) in MARKERS.items():
            sub = cond_df[cond_df["eval_construction"] == construction].sort_values("eval_verb")
            if sub.empty:
                continue
            x = np.array([x_pos[v] + x_offset for v in sub["eval_verb"]])
            y = sub["metric_mean"].to_numpy()
            ax.plot(x, y, marker, color=color, markersize=4, linestyle="none", zorder=2, label=label)

    ax.set_xticks(range(len(eval_verbs)))
    ax.set_xticklabels([fr"$\textit{{{verb}}}$" for verb in eval_verbs])
    ax.set_xlabel("Verb")
    ax.set_ylabel(f"Mean {metric_label}")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    condition_handles = [
        mlines.Line2D([], [], color=color, marker="o", markersize=4, label=label) for cond, label, color in CONDITIONS
    ]
    marker_handles = [
        mlines.Line2D([], [], color="gray", marker=marker, markersize=4, linestyle="none", label=label)
        for construction, (label, marker) in MARKERS.items()
    ]
    ax.legend(
        handles=condition_handles + marker_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=2,
        frameon=False,
    )
    # context_title = context_label.replace("_", r"\_")
    # ax.set_title(f"Causative vs. transitive")
    despine(ax)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset-dir", type=Path, default=TESTSET_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--context",
        choices=["with_context", "no_context", "both"],
        default="both",
    )
    args = parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    context_flags: list[tuple[bool, str]] = []
    if args.context in ("with_context", "both"):
        context_flags.append((True, "with_context"))
    if args.context in ("no_context", "both"):
        context_flags.append((False, "no_context"))

    for has_context, context_label in context_flags:
        print(f"Loading data ({context_label})...")
        df = load_data(args.testset_dir, has_context)
        print(
            f"  {len(df):,} rows  |  {df['eval_verb'].n_unique()} verbs  |  conditions: {sorted(df['condition'].unique())}"
        )

        for metric_key, (metric_col, metric_label) in METRICS.items():
            agg = aggregate(df, metric_col)
            out_path = args.figure_dir / f"20260512.preemption-two-way-dumbbell_{context_label}_{metric_key}.pdf"
            plot_dumbbell(agg, metric_label, context_label, out_path)


if __name__ == "__main__":
    main()
