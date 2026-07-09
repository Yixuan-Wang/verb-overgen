"""
Count V-overlap bigram strength for each overgen test-set sentence.

A V-overlap bigram is a test-set bigram that also appears in the training
bigram counts, has the row's target verb as either token, and has neither token
in the excluded set: <s>, </s>, tokens containing ASCII punctuation, or "the".
The sentence-level count is the maximum training-set count among matching
V-overlap bigrams, or 0 if there is no V-overlap bigram.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import string
from typing import cast

from omegaconf import OmegaConf
import polars as pl
from tokenizers import Tokenizer

DEFAULT_CONFIG_PATH = Path("./src/verb_overgen/train/configs/train.yaml")
DEFAULT_BIGRAM_COUNT_PATH = Path("./var/pychildes/ngram/bigram_count.parquet")
DEFAULT_OUTPUT_PATH = Path("./var/testset/testset_bigram_overlap.parquet")


def path_from_config(config_path: Path, key: str) -> Path:
    cfg = OmegaConf.load(config_path)
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"No {key!r} found in {config_path}")
    return Path(str(value))


def excluded_token_ids(tokenizer: Tokenizer) -> list[int]:
    punctuation = set(string.punctuation)
    return [
        token_id
        for token, token_id in tokenizer.get_vocab().items()
        if token in {"<s>", "</s>", "the"} or any(char in punctuation for char in token)
    ]


def verb_token_ids(tokenizer: Tokenizer, verbs: list[str]) -> pl.DataFrame:
    rows: list[dict[str, int | str]] = []
    missing_verbs: list[str] = []
    for verb in verbs:
        token_id = tokenizer.token_to_id(verb)
        if token_id is None:
            missing_verbs.append(verb)
            continue
        rows.append({"verb": verb, "verb_token_id": token_id})

    if missing_verbs:
        missing = ", ".join(sorted(missing_verbs))
        raise ValueError(f"These verbs are missing from the tokenizer vocabulary: {missing}")

    return pl.DataFrame(rows, schema={"verb": pl.String, "verb_token_id": pl.Int64})


def count_v_bigram_overlap(
    *,
    testset_path: Path,
    bigram_count_path: Path,
    tokenizer_path: Path,
) -> pl.DataFrame:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    df_testset = pl.scan_parquet(testset_path).filter(~pl.col("kind").str.starts_with("sentence_context"))
    df_bigram_count = pl.scan_parquet(bigram_count_path)

    verbs = df_testset.select("verb").unique().collect()["verb"].to_list()
    df_verb_token_ids = verb_token_ids(tokenizer, verbs).lazy()
    excluded_ids = excluded_token_ids(tokenizer)

    df_testset_bigrams = (
        df_testset.with_row_index("index_testset")
        .select(
            "index_testset",
            "index_sentence",
            "verb",
            "kind",
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

    df_v_overlap_bigram = (
        df_testset_bigrams.join(
            df_bigram_count,
            on=["token-1", "token-2"],
            how="left",
            validate="m:1",
        )
        .join(df_verb_token_ids, on="verb", how="left", validate="m:1")
        .with_columns(pl.col("count").fill_null(0).cast(pl.UInt32).alias("count_train"))
        .with_columns(
            (
                (pl.col("count_train") > 0)
                & ~pl.col("token-1").is_in(excluded_ids)
                & ~pl.col("token-2").is_in(excluded_ids)
                & (
                    (pl.col("token-1") == pl.col("verb_token_id"))
                    | (pl.col("token-2") == pl.col("verb_token_id"))
                )
            ).alias("is_v_bigram_overlap")
        )
    )

    return cast(
        pl.DataFrame,
        df_v_overlap_bigram.group_by(
            "index_testset",
            "index_sentence",
            "verb",
            "kind",
            maintain_order=True,
        )
        .agg(
            pl.when(pl.col("is_v_bigram_overlap"))
            .then(pl.col("count_train"))
            .otherwise(0)
            .max()
            .cast(pl.UInt32)
            .alias("count_v_bigram_overlap")
        )
        .drop("index_testset")
        .sort("index_sentence", "kind")
        .collect(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--bigram-count", type=Path, default=DEFAULT_BIGRAM_COUNT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    testset_path = path_from_config(args.config, "data.eval_overgen.path")
    tokenizer_path = path_from_config(args.config, "tokenizer")
    df_overlap = count_v_bigram_overlap(
        testset_path=testset_path,
        bigram_count_path=args.bigram_count,
        tokenizer_path=tokenizer_path,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_overlap.write_parquet(args.output)
    print(f"Wrote {len(df_overlap):,} sentence rows to {args.output}")


if __name__ == "__main__":
    main()
