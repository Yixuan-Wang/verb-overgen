"""Extract features from PyChildes data and mark them within the dataset."""
from pathlib import Path

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict
import polars as pl

from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_verb, extract_periphrastic_make
from verb_overgen.prep.lib.syntactic import Parse, ParseWithPos
from verb_overgen.prep.lib.lexical import LIST_CAUSATIVE_VERBS, LIST_EXTENDED_CAUSATIVE_VERBS
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

    if "parse" not in dataset.collect_schema().names():
        dataset = dataset.with_columns(
            pl.struct(
                pl.col("syntax").struct.field("to_index").alias("arc"),
                pl.col("syntax").struct.field("rel").alias("rel"),
                pl.col("morph").struct.field("lemma").alias("word"),
            ).alias("parse")
        )
        has_changed = True

    return dataset, has_changed


set_base_causative_verbs = set(LIST_CAUSATIVE_VERBS)
set_extended_causative_verbs = set(LIST_EXTENDED_CAUSATIVE_VERBS())
set_all_causative_verbs = set_base_causative_verbs.union(set_extended_causative_verbs)

class Features(TypedDict):
    features: list[str]
    featured_verbs: list[dict[str, str]]


def mark_causative(
    parse: ParseWithPos,
    # pos: pl.Expr,
) -> Features:
    # Extract all causative verbs, supported by "make" and "let".
    list_extracted_make_causative_verb = extract_periphrastic_verb(
        parse=parse,
        support_verb="make",
        verify_subj=False,
    )
    # list_extracted_let_causative_verb = extract_periphrastic_verb(
    #     parse=parse,
    #     support_verb="let",
    #     verify_subj=True,
    # )

    features: set[str] = set()
    """Gradually build up the features set."""

    featured_verbs: list[dict[str, str]] = []
    """Gradually build up the featured verbs list."""
    
    set_featured_causative_verbs: set[str] = set()
    for support_verb, extracted in zip(
        ["make",], # "let"],
        [list_extracted_make_causative_verb, ], # list_extracted_let_causative_verb],
    ):
        if extracted:
            set_extracted = set(extracted)
            set_this_causative_verbs = set_extracted.intersection(set_all_causative_verbs)
            if set_this_causative_verbs:
                features.add(FEATURE_CAUSATIVE)
                set_featured_causative_verbs.update(set_this_causative_verbs)
            for verb in set_this_causative_verbs:
                featured_verbs.append(
                    {
                        "feature": FEATURE_CAUSATIVE,
                        "verb": verb,
                        "extra": json.dumps({
                            "support_verb": support_verb,
                            "is_base": verb in set_base_causative_verbs,
                        }),
                    }
                )
    
    set_entrenched_causative_verbs = (
        set(parse["word"]).intersection(set_all_causative_verbs).difference(set_featured_causative_verbs)
    )
    if set_entrenched_causative_verbs:
        features.add(FEATURE_CAUSATIVE_ENTRENCHED)
    for verb in set_entrenched_causative_verbs:
        featured_verbs.append(
            {
                "feature": FEATURE_CAUSATIVE_ENTRENCHED,
                "verb": verb,
                "extra": json.dumps({
                    "support_verb": None,
                    "is_base": verb in set_base_causative_verbs,
                }),
            }
        )

    return {
        "features": list(features),
        "featured_verbs": featured_verbs,
    }


def old_logic():
    dataset, has_changed = load_data("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")
    OUTPUT_PATH = "./data/pychildes/pychildes-morphosyntax_annotation-20260225.20251024.parquet"
    
    if Path(OUTPUT_PATH).exists():
        print(f"Output file {OUTPUT_PATH} already exists. Please remove it before running the script.")
        exit(1)
    
    
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

    dataset_annotation.sink_parquet(OUTPUT_PATH)
    elapsed = time.perf_counter() - t0
    print(f"mark_causative took {elapsed:.3f}s")

def debug():
    from verb_overgen.prep.lib.pychildes_rule_causative import RuleLet

    dataset, has_changed = load_data("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")
    # OUTPUT_PATH = "./data/pychildes/pychildes-morphosyntax_annotation-20260225.20251024.parquet"
    # dataset = dataset.head(10000)
    dataset_let_verb = RuleLet.find(dataset)
    
    __import__("IPython").embed(using=False)


if __name__ == "__main__":
    debug()
