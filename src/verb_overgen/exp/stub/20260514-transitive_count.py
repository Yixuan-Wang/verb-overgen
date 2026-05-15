"""Count transitive usages of the 10 verbs of interest in the PyChildes parquet.

A "transitive usage" is an occurrence of a target verb (matched on lemma) that has
at least one `OBJ` child in the dependency parse.
"""

import os

os.chdir(os.environ["PWD"])

import polars as pl

from verb_overgen.common.features import VERB_TARGET_10
from verb_overgen.prep.lib.syntactic import Parse

DATA_PATH = "data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"

VERBS_OF_INTEREST = ["rise", "smile", "blink", "sing", "sneeze", "sleep", "cry", "laugh", "look", "go"]


def count_transitive_per_verb(parse: Parse, *, target: str) -> int:
    """Number of occurrences of `target` (by lemma) that have an `OBJ` child."""
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

    if target not in list_word:
        return 0
    if "OBJ" not in list_rel:
        return 0

    count = 0
    for index1, word in enumerate(list_word, 1):
        if word != target:
            continue
        if any(arc == index1 and rel == "OBJ" for arc, rel in zip(list_arc, list_rel)):
            count += 1
    return count


def main() -> None:
    dataset = pl.scan_parquet(DATA_PATH)

    parse_expr = pl.struct(
        pl.col("syntax").struct.field("to_index").alias("arc"),
        pl.col("syntax").struct.field("rel").alias("rel"),
        pl.col("morph").struct.field("lemma").alias("word"),
    )

    df = (
        dataset.filter(pl.col("has_parse"))
        .select(parse=parse_expr)
        .with_columns(
            pl.col("parse")
            .map_elements(
                lambda parse: {
                    verb: count_transitive_per_verb(parse, target=verb)
                    for verb in VERBS_OF_INTEREST
                },
                return_dtype=pl.Struct({verb: pl.Int64 for verb in VERBS_OF_INTEREST}),
            )
            .alias("counts")
        )
        .select(pl.col("counts").struct.field(verb).sum().alias(verb) for verb in VERBS_OF_INTEREST)
        .collect()
    )

    counts = {verb: df[0, verb] for verb in VERBS_OF_INTEREST}
    total = sum(counts.values())

    print("Transitive usages (verb has an OBJ child):")
    for verb in sorted(VERBS_OF_INTEREST, key=lambda v: counts[v], reverse=True):
        print(f"  {verb:8s} {counts[verb]:6d}")
    print(f"  {'TOTAL':8s} {total:6d}")


if __name__ == "__main__":
    main()
