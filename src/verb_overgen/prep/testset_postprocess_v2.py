"""Merge different SCAMP data into a single table."""

from pathlib import Path
from typing import cast

import polars as pl
import pyarrow as pa
from loguru import logger

from verb_overgen.common.tokenize import UNK_TOKEN
from verb_overgen.prep.lib.lexical import BaseSpacyParser
from verb_overgen.prep.lib.testset_rule import extract_causative_sentences, sha256_hash


def load_vetted_data(
    input_path: str | Path,
) -> pl.DataFrame:
    """Load the vetted testset data from a CSV file."""
    df = pl.read_csv(input_path, separator="\t")

    df = (
        df.lazy()
        .select(
            pl.col("verb", "hash_context"),
            pl.all_horizontal(pl.selectors.starts_with("vet").eq("no").not_()).alias("verdict"),
            pl.selectors.starts_with("sentence"),
        )
        .filter(pl.col("verdict"))
        .select(
            pl.col("verb"),
            pl.row_index("index_verb_context").over("verb"),
            pl.col("hash_context"),
            pl.selectors.starts_with("sentence"),
        )
        .collect()
    )

    return cast(pl.DataFrame, df)


def load_data(*input_path: str | Path) -> pl.DataFrame:
    if len(input_path) == 0:
        raise ValueError("At least one input path must be provided.")
    elif len(input_path) == 1:
        df = pl.read_parquet(input_path[0])
    else:
        df = pl.concat([pl.read_parquet(path) for path in input_path], how="vertical")

    expr_error = pl.col("llm_gen").struct["error"].is_not_null() | pl.col("llm_gen").struct["output_text"].is_null()
    error = df.select(expr_error.sum()).item()
    if error > 0:
        logger.warning(f"In {input_path} found number of rows with error: {error}")

    df = (
        df.lazy()
        .filter(expr_error.not_())
        .select(
            pl.col("verb"),
            pl.row_index("index_verb_context").over("verb"),
            pl.col("llm_gen").struct["output_text"],
        )
        .select(
            pl.col("verb", "index_verb_context"),
            sha256_hash(
                pl.col("verb"),
                pl.col("index_verb_context").cast(pl.String),
                pl.col("output_text"),
            ).alias("hash_context"),
            extract_causative_sentences(
                pl.col("output_text"),
                verb=pl.col("verb"),
            )
            .struct.unnest()
            .name.prefix("sentence_"),
        )
        .collect()
    )

    return cast(pl.DataFrame, df)


class SpacyParser(BaseSpacyParser):
    """A Ray data batch mapper that adds Spacy parses to testset dataset."""

    def __call__(self, table: pa.Table, input_column: str = "sentence", output_column: str = "roberta") -> pa.Table:
        sentence = table.column(input_column)
        arr_roberta = self.run_spacy(sentences=sentence.to_pylist())
        return table.append_column(output_column, arr_roberta)


if __name__ == "__main__":
    DATASET_RAW = [
        load_vetted_data("./var/testset/overgen-gemini-voi_5-vet_v1-20260312T045725.csv"),
        load_data("./var/testset/overgen-gemini-voi_12-20260316T143434.parquet"),
        load_data(
            "./var/testset/overgen-gemini-voi_extension_0326-20260326T193113.parquet",
            "./var/testset/overgen-gemini-voi_extension_0326_residual-20260402T195251.parquet",
        ),
    ]

    dataset_raw = pl.concat(DATASET_RAW)

    is_sentence_unique = dataset_raw.select(
        pl.concat_str("sentence_context", "sentence_causative", separator="\x00").is_unique().all()
    ).item()

    count_causative_sentence_unique = dataset_raw.select(pl.col("sentence_causative").is_duplicated().sum()).item()

    if not is_sentence_unique:
        raise NotImplementedError("The testset contains duplicate. Deduplication is not implemented yet.")

    if count_causative_sentence_unique > 0:
        logger.warning(f"The testset contains {count_causative_sentence_unique} duplicate causative sentences.")

    dataset_long = dataset_raw.unpivot(
        on=pl.selectors.starts_with("sentence_"),
        index=~pl.selectors.starts_with("sentence_"),
        variable_name="kind",
        value_name="sentence",
    )

    dataset_roberta = dataset_long.pipe(lambda df: cast(pl.DataFrame, pl.from_arrow(SpacyParser()(df.to_arrow()))))

    # * How to pivot back
    # dataset_roberta.pipe(lambda df: df.pivot(
    #     on="kind",
    #     values=pl.selectors.by_index(range(df.get_column_index("kind") + 1, df.width)),
    #     index=pl.selectors.by_index(range(0, df.get_column_index("kind")))
    # ))

    import tokenizers as hftk

    tokenizer_file = "./data/tokenizer/1024-wordlevel/tokenizer.json"
    tokenizer = hftk.Tokenizer.from_file(tokenizer_file)

    from verb_overgen.train.run_tokenize import tokenize as _tokenize

    def tokenize(sentences):
        input_ids = _tokenize(tokenizer, sentences)
        return pl.Series("input_ids", input_ids, dtype=pl.List(pl.Int64))

    dataset_tokenized = dataset_roberta.with_columns(
        pl.col("roberta")
        .struct.field("lemma")
        .list.join(" ")
        .map_batches(tokenize, pl.List(pl.Int64))
        .alias("input_ids")
    )

    def concat_roberta(lhs: pl.Expr, rhs: pl.Expr):
        return pl.struct(
            pl.concat_list(lhs.struct["lemma"], rhs.struct["lemma"]).alias("lemma"),
            pl.concat_list(lhs.struct["morph"], rhs.struct["morph"]).alias("morph"),
            pl.lit([], pl.List(pl.Int32)).alias("dep_arc"),
            pl.lit([], pl.List(pl.String)).alias("dep_rel"),
            pl.concat_list(lhs.struct["pos_ud"], rhs.struct["pos_ud"]).alias("pos_ud"),
            pl.concat_list(lhs.struct["pos_penn"], rhs.struct["pos_penn"]).alias("pos_penn"),
        )

    def concat_input_ids(lhs: pl.Expr, rhs: pl.Expr):
        return pl.concat_list(lhs.list.slice(0, lhs.list.len() - 1), rhs)

    dataset_tokenized_wide = dataset_tokenized.pipe(
        lambda df: df.pivot(
            on="kind",
            values=["sentence", "roberta", "input_ids"],
            index="hash_context",
        )
    )
    dataset_tokenized_wide = dataset_raw.select("verb", "index_verb_context", "hash_context").join(
        dataset_tokenized_wide, on=["hash_context"], how="left", validate="1:1"
    )

    expr_no_unk = ~pl.any_horizontal(
        pl.selectors.starts_with("input_ids").list.contains(tokenizer.token_to_id(UNK_TOKEN))
    )
    dataset_tokenized_wide = dataset_tokenized_wide.filter(
        expr_no_unk,
    )

    dataset_tokenized_long_context_causative = dataset_tokenized_wide.select(
        "verb",
        "index_verb_context",
        "hash_context",
        pl.lit("sentence_context_causative").alias("kind"),
        pl.concat_str("sentence_sentence_context", pl.lit(" "), "sentence_sentence_causative").alias("sentence"),
        concat_roberta(pl.col("roberta_sentence_context"), pl.col("roberta_sentence_causative")).alias("roberta"),
        concat_input_ids(pl.col("input_ids_sentence_context"), pl.col("input_ids_sentence_causative")).alias(
            "input_ids"
        ),
    )
    dataset_tokenized_long_context_transitive = dataset_tokenized_wide.select(
        "verb",
        "index_verb_context",
        "hash_context",
        pl.lit("sentence_context_transitive").alias("kind"),
        pl.concat_str("sentence_sentence_context", pl.lit(" "), "sentence_sentence_transitive").alias("sentence"),
        concat_roberta(pl.col("roberta_sentence_context"), pl.col("roberta_sentence_transitive")).alias("roberta"),
        concat_input_ids(pl.col("input_ids_sentence_context"), pl.col("input_ids_sentence_transitive")).alias(
            "input_ids"
        ),
    )

    dataset_tokenized_long_no_context = dataset_tokenized.filter(pl.col("kind") != "sentence_context").join(
        dataset_tokenized_wide.select("hash_context"),
        on="hash_context",
        how="semi",
    )
    dataset_tokenized_long = pl.concat(
        [
            dataset_tokenized_long_no_context,
            dataset_tokenized_long_context_causative,
            dataset_tokenized_long_context_transitive,
        ]
    )

    assert (
        dataset_tokenized_long.group_by("verb", "index_verb_context")
        .agg(pl.len() == 4)
        .select(pl.col("len").all())
        .item()
    ), (
        "Each verb context should have 4 sentences (context, causative, transitive, and one of the non-context sentences)."
    )

    dataset_final = dataset_tokenized_long
    __import__("ipdb").set_trace()

    OUTPUT_FILE_VER = "20260402T2045"
    path_output = Path(f"./data/testset/output/testset_gemini.{OUTPUT_FILE_VER}.parquet")

    path_output_suffix = 0
    while path_output.exists():
        path_output_suffix += 1
        path_output = Path(f"./data/testset/output/testset_gemini.{OUTPUT_FILE_VER}_{path_output_suffix}.parquet")

    dataset_final.with_row_index("index_sentence").write_parquet(path_output)
    logger.success(f"Saved the final dataset to {path_output}")
