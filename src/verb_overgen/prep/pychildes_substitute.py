"""Substitute causative sentences with transitive ones and tokenize."""

import os
from typing import cast

import polars as pl
import tokenizers as hftk
from loguru import logger

from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.common.tokenize import UNK_TOKEN
from verb_overgen.prep.lib.pychildes_rule_causative import substitute_causative_with_transitive
from verb_overgen.train.run_tokenize import tokenize as _tokenize

os.chdir(os.environ.get("PWD", "."))

dataset_main = pl.scan_parquet("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")
dataset_annotation = pl.scan_parquet(
    "./data/pychildes/pychildes-morphosyntax_annotation-20260402.20251024.parquet"
)

dataset = cast(
    pl.DataFrame,
    dataset_main.join(dataset_annotation, on=["id_file", "id_utterance"])
    .filter(pl.col("features").list.contains(FEATURE_CAUSATIVE))
    .collect(),
)

dataset = dataset.with_columns(
    pl.col("featured_verbs").list.filter(
        pl.element().struct["feature"] == FEATURE_CAUSATIVE
    ).list.eval(
        pl.element().struct["verb"]
    ).list.unique().alias("causative_verbs")
)

def _substitute(row):
    sentence, target = row["sentence"], row["causative_verbs"]
    for verb in target:
        sentence = substitute_causative_with_transitive(sentence, verb)
    return sentence

dataset = dataset.with_columns(
    pl.struct("sentence", "causative_verbs")
    .map_elements(
        _substitute,
        return_dtype=pl.String,
    )
    # Override the original sentence with the substituted one
    .alias("sentence")
)

tokenizer_file = "./data/tokenizer/1024-wordlevel/tokenizer.json"
tokenizer = hftk.Tokenizer.from_file(tokenizer_file)

input_ids = _tokenize(tokenizer, dataset["sentence"].to_list())

dataset_final = (
    dataset.select(
        "id_file",
        "id_utterance",
        pl.col("causative_verbs").alias("_causative_verbs"),
        "sentence",
        "content_raw",
    )
    .with_columns(pl.Series("input_ids", input_ids, dtype=pl.List(pl.Int64)))
)

logger.info(f"Total rows: {len(dataset_final)}")

if __name__ == "__main__":
    from pathlib import Path

    output_path = Path("./var/pychildes/substitute/pychildes-causative_substitute_transitive.20260517.parquet")
    dataset_final.write_parquet(output_path)
    logger.success(f"Saved to {output_path}")
