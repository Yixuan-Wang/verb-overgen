"""
Extract bigram statistics from input_ids.
"""

from typing import cast
import polars as pl

DATASET_PATH = "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"

df_all = pl.scan_parquet(DATASET_PATH)

df_bigram_count = cast(
    pl.DataFrame,
    df_all
    .select(
        pl.col("input_ids").alias("token-1"),
        pl.col("input_ids").list.shift(-1).alias("token-2"),
    )
    .explode("token-1", "token-2")
    .drop_nulls()
    .group_by("token-1", "token-2")
    .len()
    .rename({"len": "count"})
    .sort("token-1", "token-2")
    .collect()
)

df_bigram_count.write_parquet("./var/pychildes/ngram/bigram_count.parquet")
