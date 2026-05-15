"""Filter out all utterances where a given verb has an `OBJ` child, then drop into IPython.

Usage:
    python src/verb_overgen/exp/stub/20260514-transitive_inspect.py [verb]
"""

import os
import sys

os.chdir(os.environ["PWD"])

import polars as pl

from verb_overgen.prep.lib.syntactic import Parse

DATA_PATH = "data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"

DEFAULT_VERB = "dance"


def has_obj_child(parse: Parse, *, target: str) -> bool:
    """True if any occurrence of `target` (by lemma) has an `OBJ` child."""
    list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

    if target not in list_word:
        return False
    if "OBJ" not in list_rel:
        return False

    return any(
        arc == index1 and rel == "OBJ"
        for index1, word in enumerate(list_word, 1)
        if word == target
        for arc, rel in zip(list_arc, list_rel)
    )


def main() -> None:
    verb = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VERB

    parse_expr = pl.struct(
        pl.col("syntax").struct.field("to_index").alias("arc"),
        pl.col("syntax").struct.field("rel").alias("rel"),
        pl.col("morph").struct.field("lemma").alias("word"),
    )

    df = (
        pl.scan_parquet(DATA_PATH)
        .filter(pl.col("has_parse"))
        .with_columns(parse_expr.alias("parse"))
        .filter(
            pl.col("parse").map_elements(
                lambda parse: has_obj_child(parse, target=verb),
                return_dtype=pl.Boolean,
            )
        )
        .collect()
    )

    print(f"verb = {verb!r}: {df.height} utterances with an OBJ-child occurrence")
    print("DataFrame available as `df`.")

    import IPython

    IPython.embed()


if __name__ == "__main__":
    main()
