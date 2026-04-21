"""
Extract unigram statistics from the original non-lemmatized sentences.
"""


import string

import polars as pl

PUNCT = string.punctuation

DATASET_PATH = "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"

df_all = pl.scan_parquet(DATASET_PATH)

df_unigram_count = df_all.select(pl.col("content_raw").str.split(" ").alias("unigram").list.explode().value_counts()).collect()

df_unigram_count = df_unigram_count.select(pl.col("unigram").struct.unnest()).sort("count", descending=True)

_stripped = pl.col("unigram").str.strip_chars(PUNCT)
df_unigram_stripped_count = (
    df_unigram_count.with_columns(
        pl.when(_stripped.str.len_chars() == 0)
        .then(pl.col("unigram"))
        .otherwise(_stripped)
        .alias("unigram_stripped")
    )
    .group_by("unigram_stripped")
    .agg(pl.col("count").sum())
    .sort("count", descending=True)
)

__import__("IPython").embed(using=False)
