"""Merge different SCAMP data into a single table."""

from collections.abc import Iterable
from functools import partial

import ipdb
import pyarrow as pa
import polars as pl
import regex as re

from typing import cast

from verb_overgen.prep.lib.lexical import BaseSpacyParser
from verb_overgen.prep.lib.testset_rule import alternate_transitive

def load_dataset(
    input_path: str,
    *,
    sample_n: int = 3,
    sample_seed: int = 42,
) -> pl.DataFrame:
    dataset = pl.read_parquet(input_path)
    dataset = dataset.select(
        pl.all().exclude("fragment_object", "fragment_noun_bank"),
    )

    return dataset

def extract_sentences(
    output_text: pl.Expr,
    *,
    verb: pl.Expr,
):
    """
    Examples:
    ```
    dataset.select(
        extract_sentences(
            pl.col("llm_gen").struct.field("output_text"),
            verb=pl.col("verb")
        )
    )
    ```
    """
    output_text = output_text.str.split("\n")
    sentence_context = output_text.list.first().str.strip_chars().alias("context")
    sentence_causative = output_text.list.get(1).str.strip_chars().alias("causative")
    sentence_transitive = pl.struct(sentence_causative, verb).map_elements(lambda s: alternate_transitive(s["causative"], s["verb"])).alias("transitive")
    return pl.struct(sentence_context, sentence_causative, sentence_transitive).alias("sentence")

class SpacyParser(BaseSpacyParser):
    """A Ray data batch mapper that adds Spacy parses to testset dataset."""

    def __call__(self, table: pa.Table, input_column: str = "sentence", output_column: str = "roberta") -> pa.Table:
        sentence = table.column(input_column)
        arr_roberta = self.run_spacy(sentences=sentence.to_pylist())
        return table.append_column(output_column, arr_roberta)

# def unpivot_dataset(df: pl.DataFrame) -> pl.DataFrame:
#     return df.unpivot(
#         on=pl.selectors.starts_with("sentence_"),
#         index="index",
#         variable_name="kind",
#         value_name="sentence",
#     )


# def pivot_dataset(
#     df_roberta: pl.DataFrame,
# ) -> pl.DataFrame:
#     return df_roberta.pivot(on="kind", index="index", values=["sentence", "roberta"]).select(
#         pl.all().name.replace("_sentence_", "_")
#     )


# if __name__ == "__main__":
#     dataset = load_dataset("data/testset/output/testset_llm_gen_supplementary.20260108T2325.parquet")
#     dataset_long = unpivot_dataset(dataset)

#     parser = SpacyParser()
#     dataset_roberta = cast(pl.DataFrame, pl.from_arrow(parser(dataset_long.to_arrow())))
#     dataset_roberta_wide = pivot_dataset(dataset_roberta)

#     dataset = dataset.join(dataset_roberta_wide, on="index", coalesce=True)

#     dataset.write_parquet("./data/testset/output/testset_llm_gen_supplementary-postprocessed.20260108T2325.parquet")

if __name__ == "__main__":
    from verb_overgen.prep.testset_vet import load_data_vetted, load_data_original, vet

    INPUT_PATH_VETTED = "./var/testset/verb-overgen_testset_vetted_20260114T1637.csv"
    INPUT_PATH_ORIGINAL = "./data/testset/output/testset_llm_gen.final.20260108T2325.parquet"

    dataset_vetted = load_data_vetted(INPUT_PATH_VETTED)
    dataset_original = load_data_original(INPUT_PATH_ORIGINAL)

    dataset = vet(
        dataset_vetted,
        dataset_original,
    )

    target_verbs = ("laugh", "cry", "sing", "smile", "sleep")
    vet_rule = pl.col("verb").is_in(target_verbs).and_(
        pl.col("vet").struct.field("verdict_kanishka").is_null().or_(pl.col("vet").struct.field("verdict_kanishka") == 1).and_(
            pl.col("vet").struct.field("verdict_tom").is_null().or_(pl.col("vet").struct.field("verdict_tom") == 1),
        )
    )

    dataset = dataset.with_columns(vet_rule.alias("vet"))

    dataset_sentence = dataset.select(
        "index",
        "verb",
        "vet",
        extract_sentences(
            pl.col("llm_gen").struct.field("output_text"),
            verb=pl.col("verb")
        )
    )
    
    dataset_long = (
        dataset_sentence
            .select(
                "index",
                "verb",
                "vet",
                pl.col("sentence").struct.unnest().name.prefix("sentence_")
            )
            .unpivot(
                on=pl.selectors.starts_with("sentence_"),
                index=["index", "verb", "vet"],
                variable_name="kind",
                value_name="sentence",
            )
            .with_row_index("index_sentence")
    )

    dataset_roberta = dataset_long.pipe(lambda df: cast(pl.DataFrame, pl.from_arrow(SpacyParser()(df.to_arrow()))))

    import tokenizers as hftk
    tokenizer_file = "./data/tokenizer/1024-wordlevel/tokenizer.json"
    tokenizer = hftk.Tokenizer.from_file(tokenizer_file)


    from verb_overgen.train.run_tokenize import tokenize as _tokenize
    def tokenize(sentences):
        input_ids = _tokenize(tokenizer, sentences)
        return pl.Series("input_ids", input_ids, dtype=pl.List(pl.Int64))

    dataset_tokenized = dataset_roberta.with_columns(
        pl.col("roberta").struct.field("lemma").list.join(" ").map_batches(tokenize, pl.List(pl.Int64)).alias("input_ids")
    )

    dataset_tokenized.write_parquet("./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet")
    
