"""Merge different SCAMP data into a single table."""

from pathlib import Path
import pyarrow as pa
import polars as pl

from typing import cast

from verb_overgen.prep.lib.lexical import BaseSpacyParser

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
            pl.col("verb", "hash_context"),
            pl.row_index("index_verb_context").over("verb"),
            pl.col("verdict"),
            pl.selectors.starts_with("sentence"),
        )
        .collect()
    )

    return df

class SpacyParser(BaseSpacyParser):
    """A Ray data batch mapper that adds Spacy parses to testset dataset."""

    def __call__(self, table: pa.Table, input_column: str = "sentence", output_column: str = "roberta") -> pa.Table:
        sentence = table.column(input_column)
        arr_roberta = self.run_spacy(sentences=sentence.to_pylist())
        return table.append_column(output_column, arr_roberta)



if __name__ == "__main__":

    INPUT_PATH_VETTED = "./var/testset/overgen-gemini-voi_5-vet_v1-20260312T045725.csv"
    
    dataset_vetted = load_vetted_data(INPUT_PATH_VETTED)

    dataset_vetted = dataset_vetted.select(
        ~pl.selectors.starts_with("sentence"),
        pl.concat_str("sentence_context", pl.lit(" "), "sentence_causative").alias("sentence_context_causative"),
        pl.concat_str("sentence_context", pl.lit(" "), "sentence_transitive").alias("sentence_context_transitive"),
        "sentence_causative",
        "sentence_transitive",
    )

    dataset_long = dataset_vetted.unpivot(
        on=pl.selectors.starts_with("sentence_"),
        index=~pl.selectors.starts_with("sentence_"),
        variable_name="kind",
        value_name="sentence",
    )

    dataset_long = dataset_long.with_row_index("index_sentence")

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
        pl.col("roberta").struct.field("lemma").list.join(" ").map_batches(tokenize, pl.List(pl.Int64)).alias("input_ids")
    )

    # __import__("IPython").embed(using=False)
    dataset_tokenized.write_parquet("./data/testset/output/testset_gemini_voi-5.20260312T2040.parquet")
    
