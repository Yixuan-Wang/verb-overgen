"""Extract features from PyChildes data and mark them within the dataset."""
import json
import time
from pathlib import Path
from typing import TypedDict

import polars as pl

from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.prep.lib.lexical import LIST_CAUSATIVE_VERBS, LIST_EXTENDED_CAUSATIVE_VERBS
from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_verb
from verb_overgen.prep.lib.pychildes_rule_causative import MixinRuleFind, RuleLet, RuleMake
from verb_overgen.prep.lib.syntactic import Parse


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


MAP_PERIPHERAL_RULES: dict[str, type[MixinRuleFind]] = { rule.PERIPHERAL_VERB: rule for rule in [RuleLet, RuleMake] }

set_base_causative_verbs = set(LIST_CAUSATIVE_VERBS)
set_extended_causative_verbs = set(LIST_EXTENDED_CAUSATIVE_VERBS())
set_all_causative_verbs = set_base_causative_verbs.union(set_extended_causative_verbs)


class Features(TypedDict):
    features: list[str]
    featured_verbs: list[dict[str, str]]

# @__import__("ipdb").iex
def build_causative_features(row) -> Features:
    list_pos = row["pos"]
    list_word = row["parse"]["word"]

    # Collect extracted verbs per support verb
    map_support_verb_extracted: dict[str, list[int]] = {
        support_verb: row[f"verb_causative_{support_verb}"] or []
        for support_verb in MAP_PERIPHERAL_RULES
    }

    # verb_causative_make = extract_periphrastic_verb(
    #     parse=parse,
    #     support_verb="make",
    #     verify_subj=False,
    # )
    # if verb_causative_make:
    #     map_support_verb_extracted["make"] = verb_causative_make

    features: set[str] = set()
    featured_verbs: list[dict[str, str]] = []
    set_featured_causative_verbs: set[str] = set()

    for support_verb, list_extracted_index in map_support_verb_extracted.items():
        list_this_causative_verbs = [
            word
            for index in list_extracted_index
            if (word := list_word[index]) in set_all_causative_verbs
        ]
        if list_this_causative_verbs:
            features.add(FEATURE_CAUSATIVE)
            set_featured_causative_verbs.update(list_this_causative_verbs)
        for verb in list_this_causative_verbs:
            featured_verbs.append(
                {
                    "feature": FEATURE_CAUSATIVE,
                    "verb": verb,
                    "extra": json.dumps(
                        {
                            "support_verb": support_verb,
                            "is_base": verb in set_base_causative_verbs,
                        }
                    ),
                }
            )
            
    

    # set_entrenched_causative_verbs = set(parse["word"]) & set_all_causative_verbs - set_featured_causative_verbs
    list_entrenched_causative_verbs = [
        word
        for pos, word in zip(list_pos, list_word)
        if (word in set_all_causative_verbs and pos in {"v", "part"}) and word not in set_featured_causative_verbs
    ]
    if list_entrenched_causative_verbs:
        features.add(FEATURE_CAUSATIVE_ENTRENCHED)
    for verb in list_entrenched_causative_verbs:
        featured_verbs.append(
            {
                "feature": FEATURE_CAUSATIVE_ENTRENCHED,
                "verb": verb,
                "extra": json.dumps(
                    {
                        "support_verb": None,
                        "is_base": verb in set_base_causative_verbs,
                    }
                ),
            }
        )

    return {
        "features": list(features),
        "featured_verbs": featured_verbs,
    }


DTYPE_FEATURES = pl.Struct(
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
)


def annotate_causative(dataset: pl.LazyFrame) -> pl.LazyFrame:
    # Run each rule that operates at the dataframe level
    rule_results: dict[str, pl.DataFrame] = {
        name: rule.find(dataset)
        for name, rule in MAP_PERIPHERAL_RULES.items()
    }

    # Left-join all rule results back
    annotated = dataset
    for name, df_rule in rule_results.items():
        annotated = annotated.join(
            df_rule.lazy(),
            on=["id_file", "id_utterance"],
            how="left",
        )

    # Build features/featured_verbs from joined verb columns + parse
    verb_cols = [f"verb_causative_{name}" for name in rule_results]

    annotated = annotated.select(
        pl.selectors.starts_with("id_"),
        pl.struct("parse", pl.col("morph").struct["pos"], *verb_cols)
        .map_elements(
            build_causative_features,
            DTYPE_FEATURES,
        )
        .struct.unnest(),
    )

    return annotated


def main():
    dataset, has_changed = load_data("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")
    OUTPUT_PATH = "./var/pychildes/debug-annotation/pychildes-morphosyntax_annotation_make-let_20260319.20251024.parquet"

    if Path(OUTPUT_PATH).exists():
        print(f"Output file {OUTPUT_PATH} already exists. Please remove it before running the script.")
        exit(1)

    t0 = time.perf_counter()
    dataset_annotation = annotate_causative(dataset)
    dataset_annotation = dataset_annotation.filter(pl.col("features").list.len() > 0)
    dataset_annotation.sink_parquet(OUTPUT_PATH)
    elapsed = time.perf_counter() - t0
    print(f"annotate_causative took {elapsed:.3f}s")


if __name__ == "__main__":
    main()
