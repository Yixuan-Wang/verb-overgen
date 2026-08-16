"""Print a Markdown table of the verb-category preemption contrast, no figure.

For the two most representative verb groups — the 6 original intransitive-only
verbs (`VERB_TARGET_FINALE`) and the 7 new alternating verbs
(`VERB_TARGET_EXTENSION["extended_alternating"]`) — contrasts
`remove::e::<scope>` against `remove::p::<scope>` on `transitive :: no_context`
NLL at the final training epoch, per eval verb, for both `all-verb` and
`all-intrans-only` scopes side by side.

The original group's `all-verb` arm comes from `var/results/v2/testset`, its
`all-intrans-only` arm from `var/results/v2/testset_intransitive-only`. The
alternating group draws both scopes from
`var/results/v2/testset_template_alternating` (both tags were collected into
the same directory). NLL-only (alternating testset has no SLOR) and
`no_context`-only (alternating testset has no `with_context` items).

Preemption hypothesis: removing the preempting causatives (p) should increase
overgeneralization more than removing entrenchment-minus-preemption (e).
Expected direction of the contrast (remove::e − remove::p): NLL > 0.
"""

import argparse
import os
from pathlib import Path

import polars as pl

from verb_overgen.common.features import VERB_TARGET_EXTENSION, VERB_TARGET_FINALE
from verb_overgen.util.r_arrow_bridge import RScript

TESTSET_DIR = Path("var/results/v2/testset")
TESTSET_INTRANS_ONLY_DIR = Path("var/results/v2/testset_intransitive-only")
TESTSET_ALTERNATING_DIR = Path("var/results/v2/testset_template_alternating")
VERB_STATS_PATH = Path("var/results/verb_stats.parquet")

VERB_TARGET_ALTERNATING = VERB_TARGET_EXTENSION["extended_alternating"]

SCOPES = ["all-verb", "all-intrans-only"]

# scope -> source dir for the original-verb group (the alternating-verb group's
# source dir is the same for every scope, since both tags were collected into it).
ORIGINAL_GROUP_DIR_BY_SCOPE: dict[str, Path] = {
    "all-verb": TESTSET_DIR,
    "all-intrans-only": TESTSET_INTRANS_ONLY_DIR,
}

# (group key, display label, verbs)
GROUPS = [
    ("original", "Original (intransitive-only)", VERB_TARGET_FINALE),
    ("alternating", "Alternating", VERB_TARGET_ALTERNATING),
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


def load_and_contrast(scope: str) -> pl.DataFrame:
    def source_dir(group_key: str) -> Path:
        return ORIGINAL_GROUP_DIR_BY_SCOPE[scope] if group_key == "original" else TESTSET_ALTERNATING_DIR

    def load_group(group_key: str, verbs: list[str]) -> pl.DataFrame:
        df = (
            pl.scan_parquet(source_dir(group_key) / "*.parquet")
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
        assert len(df) > 0, f"No rows loaded from {source_dir(group_key)} for verbs {verbs}"
        return df

    df = pl.concat([load_group(key, verbs) for key, _, verbs in GROUPS])

    contrasts = _R_CONTRAST.call_table(
        df.rename({"result_nll": "metric"}).select(
            "run_target", "run_seed", "eval_verb", "eval_index_verb_context", "metric"
        )
    )
    return contrasts.with_columns(pl.lit(scope).alias("scope"))


def _verb_order(verbs: list[str]) -> list[str]:
    stats = pl.read_parquet(VERB_STATS_PATH).filter(pl.col("verb").is_in(verbs))
    return stats.sort("count_causative", descending=True)["verb"].to_list()


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _cell(row: dict) -> str:
    return f"{row['estimate']:.3f}{_stars(row['p.value'])} ({row['std.error']:.3f})"


def print_table(contrasts: pl.DataFrame) -> None:
    by_verb_scope = {(r["eval_verb"], r["scope"]): r for r in contrasts.iter_rows(named=True)}

    print("| Verb | All-verb Δ (SE) | All-intrans-only Δ (SE) | Sign |")
    print("|---|---|---|---|")
    for _, group_label, verbs in GROUPS:
        print(f"| **{group_label}** | | | |")
        for verb in _verb_order(verbs):
            row_all_verb = by_verb_scope[(verb, "all-verb")]
            row_intrans = by_verb_scope[(verb, "all-intrans-only")]

            sig_all_verb = row_all_verb["p.value"] < 0.05
            sig_intrans = row_intrans["p.value"] < 0.05
            sign_all_verb = row_all_verb["estimate"] > 0
            sign_intrans = row_intrans["estimate"] > 0
            if not sig_all_verb and not sig_intrans:
                sign = "n.s. both"
            elif sign_all_verb == sign_intrans:
                sign = "same"
            else:
                sign = "**flip**"

            print(f"| {verb} | {_cell(row_all_verb)} | {_cell(row_intrans)} | {sign} |")

    print()
    print("*** p<.001, ** p<.01, * p<.05. Contrast: remove::e − remove::p, NLL, "
          "transitive::no_context, final epoch. Positive = consistent with preemption prediction.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    os.chdir(os.environ.get("PWD", "."))

    contrasts = pl.concat([load_and_contrast(scope) for scope in SCOPES])
    print_table(contrasts)


if __name__ == "__main__":
    main()
