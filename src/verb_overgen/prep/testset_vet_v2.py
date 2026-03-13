"""Convert the generated testset for human inspection and vetting. This includes:
- Extracting the sentences from the generated text and putting them in separate columns.
- Added columns:
  - Hash of the verb, original index of the context within a verb and also the model original output.
  - Index of the context within a verb. This may be changed after vetting.
"""

from pathlib import Path

import polars as pl
import hashlib
from loguru import logger

from verb_overgen.prep.lib.testset_rule import alternate_transitive

def load_original_data(path_input: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(path_input)

    expr_error = pl.col("llm_gen").struct["error"].is_not_null()
    error = df.select(expr_error.sum()).item()
    logger.error(f"Number of rows with error: {error}")

    df = df.lazy().filter(expr_error.not_()).select(
        pl.row_index("index"),
        pl.col("verb"),
        pl.row_index("index_verb_context").over("verb"),
        pl.col("llm_gen").struct["output_text"],
    ).select(
        pl.col("index"),
        sha256_hash(
            pl.col("verb"),
            pl.col("index_verb_context").cast(pl.String),
            pl.col("output_text"),
        ).alias("hash_context"),
        pl.col("verb", "index_verb_context"),
        extract_sentences(
            pl.col("output_text"),
            verb=pl.col("verb"),
        ).struct.unnest().name.prefix("sentence_"),
        pl.lit("-").alias("verdict"),
        pl.lit("").alias("comments"),
    ).collect()

    return df

def sha256_hash(*expr: pl.Expr) -> pl.Expr:
    """Hash the given expressions using SHA256 and return the hash as a hexadecimal string."""
    concatenated = pl.concat_str(expr, separator="\t")
    hash_expr = concatenated.map_elements(lambda x: hashlib.sha256(x.encode()).hexdigest(), pl.String)
    return hash_expr

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
    sentence_transitive = pl.struct(sentence_causative, verb).map_elements(lambda s: alternate_transitive(s["causative"], s["verb"]), pl.String).alias("transitive")
    return pl.struct(sentence_context, sentence_causative, sentence_transitive).alias("sentence")

if __name__ == "__main__":
    path_input = Path("./var/testset/overgen-gemini-voi_5-20260312T045725.parquet")
    df = load_original_data(path_input)

    path_output = Path("./var/testset/overgen-gemini-voi_5-before_vet-20260312T045725.csv")
    df.write_csv(path_output)
    logger.success(f"Wrote output to {path_output}")
    # __import__("IPython").embed(using=False)
