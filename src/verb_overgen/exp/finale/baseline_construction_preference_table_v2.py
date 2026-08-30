"""Print a Markdown table of the baseline construction preference per verb (v2).

For the untrained-manipulation baseline (`run_manipulation == "none"`), reports
the mean word-level NLL of the periphrastic causative (`causative`, "PC") and
the overgeneralized transitive causative (`transitive`) for each eval verb,
together with their difference (PC - Transitive), grouped by the three v2 verb
categories.

v2 adds standard errors. The key structural fact is that each
`eval_index_verb_context` is a *minimal pair*: the same verb context is realized
once as a periphrastic causative and once as a transitive. The PC - Transitive
difference is therefore computed *per pair* — per (seed, verb context) — and the
standard error is taken over those paired differences (5 base-model seeds x 100
verb contexts = 500 pairs per verb) at the final training epoch. Pairing removes
between-context variance, so the paired SE is the rigorous uncertainty on the
preference and is generally much tighter than treating the two construction means
as independent.

Reported per verb:
  - PC NLL, Transitive NLL: mean +/- SE over all 500 sentences.
  - PC - Transitive: mean +/- SE over the 500 paired differences.

A negative PC - Transitive means the model prefers the periphrastic causative
(expected for intransitive-only verbs); a positive value means it prefers the
bare transitive (expected for transitive-only verbs).

Source: `var/results/v2/testset_0818_wordlevel` (`none_*.parquet`), which
evaluates the v2 verb set with word-level tokenization.
"""

import argparse
import os
from pathlib import Path

import polars as pl

from verb_overgen.common.features import (
    VERB_TARGET_CANDIDATE_V2_ALTERNATING,
    VERB_TARGET_CANDIDATE_V2_TRANSITIVE_ONLY,
    VERB_TARGET_FINALE,
)

TESTSET_DIR = Path("var/results/v2/testset_0818_wordlevel")

# (display label, verbs) in table order.
GROUPS = [
    ("Intransitive-only", VERB_TARGET_FINALE),
    ("Alternating", VERB_TARGET_CANDIDATE_V2_ALTERNATING),
    ("Transitive-only", VERB_TARGET_CANDIDATE_V2_TRANSITIVE_ONLY),
]

ALL_VERBS = [v for _, verbs in GROUPS for v in verbs]

# Identifies a single minimal pair: one (seed, verb context) yields a causative
# row and a transitive row.
PAIR_KEYS = ["eval_verb", "run_seed", "eval_index_verb_context"]


def _se(col: str) -> pl.Expr:
    """Standard error of the mean: sample std (ddof=1) / sqrt(n)."""
    return pl.col(col).std() / pl.col(col).count().sqrt()


def load_pairs() -> pl.DataFrame:
    """One row per minimal pair, with per-pair PC/Transitive NLL and their diff."""
    rows = (
        pl.scan_parquet(TESTSET_DIR / "none_*.parquet")
        .filter(
            pl.col("run_manipulation") == "none",
            pl.col("eval_construction").is_in(["causative", "transitive"]),
            pl.col("eval_verb").is_in(ALL_VERBS),
            pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
        )
        .collect()
    )
    assert len(rows) > 0, f"No rows loaded from {TESTSET_DIR}"

    pairs = rows.pivot(
        on="eval_construction",
        index=PAIR_KEYS,
        values="result_nll",
    )
    # Each context must contribute exactly one causative and one transitive.
    assert pairs.select(
        pl.col("causative").is_not_null() & pl.col("transitive").is_not_null()
    ).to_series().all(), "Incomplete minimal pair (missing a construction)"

    return pairs.with_columns(
        (pl.col("causative") - pl.col("transitive")).alias("diff")
    )


def summarize(pairs: pl.DataFrame) -> pl.DataFrame:
    """Per-verb means and standard errors over the paired observations."""
    return pairs.group_by("eval_verb").agg(
        pl.col("causative").mean().alias("pc_mean"),
        _se("causative").alias("pc_se"),
        pl.col("transitive").mean().alias("tr_mean"),
        _se("transitive").alias("tr_se"),
        pl.col("diff").mean().alias("diff_mean"),
        _se("diff").alias("diff_se"),
        pl.len().alias("n"),
    )


def build_table(summary: pl.DataFrame) -> pl.DataFrame:
    by_verb = {r["eval_verb"]: r for r in summary.iter_rows(named=True)}

    rows = []
    for group_label, verbs in GROUPS:
        ordered = sorted(verbs, key=lambda v: by_verb[v]["diff_mean"])
        for verb in ordered:
            r = by_verb[verb]
            rows.append({
                "Category": group_label,
                "Verb": verb,
                "PC NLL": f"{r['pc_mean']:.2f} ± {r['pc_se']:.2f}",
                "Transitive NLL": f"{r['tr_mean']:.2f} ± {r['tr_se']:.2f}",
                "PC − Transitive": f"{r['diff_mean']:+.2f} ± {r['diff_se']:.2f}",
                "N": r["n"],
            })
    return pl.DataFrame(rows)


def print_table(table: pl.DataFrame) -> None:
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        tbl_rows=-1,
        fmt_str_lengths=100,
    ):
        print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    print_table(build_table(summarize(load_pairs())))


if __name__ == "__main__":
    main()
