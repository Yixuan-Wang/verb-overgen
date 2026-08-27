"""Print a Markdown table of the baseline construction preference per verb.

For the untrained-manipulation baseline (`run_manipulation == "none"`), reports
the mean word-level NLL of the periphrastic causative (`causative`, "PC") and
the overgeneralized transitive causative (`transitive`) for each eval verb,
together with their difference (PC - Transitive), grouped by the three v2 verb
categories.

NLL is averaged over all testset sentences and all base-model seeds at the final
training epoch. A negative PC - Transitive means the model prefers the
periphrastic causative (expected for intransitive-only verbs); a positive value
means it prefers the bare transitive (expected for transitive-only verbs).

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


def load_mean_nll() -> pl.DataFrame:
    """Mean NLL per (eval_verb, eval_construction) over sentences and seeds."""
    df = (
        pl.scan_parquet(TESTSET_DIR / "none_*.parquet")
        .filter(
            pl.col("run_manipulation") == "none",
            pl.col("eval_construction").is_in(["causative", "transitive"]),
            pl.col("eval_verb").is_in(ALL_VERBS),
            pl.col("ckpt_epoch") == pl.col("ckpt_epoch").max(),
        )
        .group_by("eval_verb", "eval_construction")
        .agg(pl.col("result_nll").mean())
        .collect()
    )
    assert len(df) > 0, f"No rows loaded from {TESTSET_DIR}"
    return df


def build_table(mean_nll: pl.DataFrame) -> pl.DataFrame:
    wide = mean_nll.pivot(
        on="eval_construction", index="eval_verb", values="result_nll"
    ).with_columns((pl.col("causative") - pl.col("transitive")).alias("diff"))

    by_verb = {r["eval_verb"]: r for r in wide.iter_rows(named=True)}

    rows = []
    for group_label, verbs in GROUPS:
        ordered = sorted(verbs, key=lambda v: by_verb[v]["diff"])
        for verb in ordered:
            r = by_verb[verb]
            rows.append({
                "Category": group_label,
                "Verb": verb,
                "PC NLL": f"{r['causative']:.2f}",
                "Transitive NLL": f"{r['transitive']:.2f}",
                "PC − Transitive": f"{r['diff']:+.2f}",
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

    print_table(build_table(load_mean_nll()))


if __name__ == "__main__":
    main()
