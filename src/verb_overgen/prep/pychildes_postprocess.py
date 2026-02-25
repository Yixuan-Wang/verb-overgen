"""Extract features from PyChildes data and mark them within the dataset."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict
import polars as pl

from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_verb, extract_periphrastic_make
from verb_overgen.prep.lib.syntactic import Parse, ParseWithPos
from verb_overgen.prep.lib.lexical import LIST_CAUSATIVE_VERBS
from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.prep.lib.pattern_matching import extract_verb
import time
import json


def load_data(file_path: str) -> tuple[pl.LazyFrame, bool]:
    """Load data.

    Extra steps:
    - Create a `parse` column following the `[arc, rel, word]` format, if not present.
    """
    dataset = pl.scan_parquet(file_path)
    has_changed = False

    if "parse" not in dataset.columns:
        dataset = dataset.with_columns(
            pl.struct(
                pl.col("syntax").struct.field("to_index").alias("arc"),
                pl.col("syntax").struct.field("rel").alias("rel"),
                pl.col("morph").struct.field("lemma").alias("word"),
            ).alias("parse")
        )
        has_changed = True

    return dataset, has_changed


set_causative_verbs = set(LIST_CAUSATIVE_VERBS)


class Features(TypedDict):
    features: list[str]
    featured_verbs: list[dict[str, str]]


def mark_causative(
    parse: ParseWithPos,
    # pos: pl.Expr,
) -> Features:
    list_extracted_make_causative = extract_periphrastic_verb(
        parse=parse,
        support_verb="make",
        verify_subj=False,
    )

    features: list[str] = []
    featured_verbs: list[dict[str, str]] = []
    set_matched_make_causative = set(list_extracted_make_causative).intersection(set_causative_verbs)
    if set_matched_make_causative:
        features.append(FEATURE_CAUSATIVE)
    for verb in set_matched_make_causative:
        featured_verbs.append(
            {
                "feature": FEATURE_CAUSATIVE,
                "verb": verb,
            }
        )

    set_unmatched_causative_verbs = (
        set(parse["word"]).intersection(set_causative_verbs).difference(set_matched_make_causative)
    )
    if set_unmatched_causative_verbs:
        features.append(FEATURE_CAUSATIVE_ENTRENCHED)
    for verb in set_unmatched_causative_verbs:
        featured_verbs.append(
            {
                "feature": FEATURE_CAUSATIVE_ENTRENCHED,
                "verb": verb,
            }
        )

    return {
        "features": features,
        "featured_verbs": featured_verbs,
    }


if __name__ == "__main__":
    dataset, has_changed = load_data("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")

    t0 = time.perf_counter()
    dataset_annotation = dataset.select(
        pl.col("id_file", "id_utterance"),
        pl.col("parse")
        .struct.with_fields(pos=pl.col("morph").struct["pos"])
        .map_elements(
            mark_causative,
            pl.Struct(
                {
                    "features": pl.List(type_features := pl.Categorical("features")),
                    "featured_verbs": pl.List(
                        pl.Struct(
                            {
                                "feature": type_features,
                                "verb": pl.String,
                                "extra": pl.String,
                            }
                        )
                    ),
                }
            ),
        )
        .struct.unnest(),
    )

    dataset_annotation.sink_parquet("./data/pychildes/pychildes-morphosyntax_annotation-20260221.20251024.parquet")
    elapsed = time.perf_counter() - t0
    print(f"mark_causative took {elapsed:.3f}s")
