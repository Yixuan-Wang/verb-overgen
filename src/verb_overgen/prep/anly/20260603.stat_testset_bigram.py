"""
Count how often each test-set bigram appears in the original training set.
"""

from pathlib import Path
from typing import cast

from omegaconf import OmegaConf
import polars as pl

CONFIG_PATH = Path("./src/verb_overgen/train/configs/train.yaml")
BIGRAM_COUNT_PATH = Path("./var/pychildes/ngram/bigram_count.parquet")
OUTPUT_PATH = Path("./var/pychildes/ngram/testset_bigram_train_count.parquet")
OUTPUT_SENTENCE_PATH = Path("./var/pychildes/ngram/testset_sentence_bigram_train_count.parquet")


def get_testset_path(config_path: Path) -> Path:
    cfg = OmegaConf.load(config_path)
    testset_path = OmegaConf.select(cfg, "data.eval_overgen.path")
    if testset_path is None:
        raise ValueError(f"No data.eval_overgen.path found in {config_path}")
    return Path(str(testset_path))


def count_testset_bigrams(
    *,
    testset_path: Path,
    bigram_count_path: Path,
) -> pl.DataFrame:
    df_testset = pl.scan_parquet(testset_path)
    df_bigram_count = pl.scan_parquet(bigram_count_path)

    df_testset_bigrams = (
        df_testset.filter(~pl.col("kind").str.starts_with("sentence_context"))
        .with_row_index("index_testset")
        .select(
            "index_testset",
            "index_sentence",
            "verb",
            "index_verb_context",
            "hash_context",
            "kind",
            "sentence",
            pl.int_ranges(0, pl.col("input_ids").list.len()).alias("index_bigram"),
            pl.col("input_ids").alias("token-1"),
            pl.col("input_ids").list.shift(-1).alias("token-2"),
        )
        .explode("index_bigram", "token-1", "token-2")
        .drop_nulls(["token-1", "token-2"])
        .with_columns(
            pl.col("token-1").cast(pl.Int32),
            pl.col("token-2").cast(pl.Int32),
        )
    )

    return cast(
        pl.DataFrame,
        df_testset_bigrams.join(
            df_bigram_count,
            on=["token-1", "token-2"],
            how="left",
            validate="m:1",
        )
        .with_columns(pl.col("count").fill_null(0).cast(pl.UInt32).alias("count_train"))
        .drop("count")
        .sort("index_testset", "index_bigram")
        .collect(),
    )


def summarize_by_sentence(df_testset_bigram_count: pl.DataFrame) -> pl.DataFrame:
    return df_testset_bigram_count.group_by(
        "index_testset",
        "index_sentence",
        "verb",
        "index_verb_context",
        "hash_context",
        "kind",
        "sentence",
        maintain_order=True,
    ).agg(
        pl.struct("index_bigram", "token-1", "token-2", "count_train").alias("bigram_count_train"),
        pl.col("count_train").sum().alias("count_train_sum"),
        pl.col("count_train").min().alias("count_train_min"),
        pl.col("count_train").max().alias("count_train_max"),
        pl.col("count_train").eq(0).sum().alias("n_bigram_unseen_train"),
        pl.len().alias("n_bigram"),
    )


def main() -> None:
    testset_path = get_testset_path(CONFIG_PATH)

    df_testset_bigram_count = count_testset_bigrams(
        testset_path=testset_path,
        bigram_count_path=BIGRAM_COUNT_PATH,
    )
    df_testset_sentence_count = summarize_by_sentence(df_testset_bigram_count)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_testset_bigram_count.write_parquet(OUTPUT_PATH)
    df_testset_sentence_count.write_parquet(OUTPUT_SENTENCE_PATH)

    print(f"Wrote {len(df_testset_bigram_count):,} test-set bigram rows to {OUTPUT_PATH}")
    print(f"Wrote {len(df_testset_sentence_count):,} test-set sentence rows to {OUTPUT_SENTENCE_PATH}")


if __name__ == "__main__":
    main()
