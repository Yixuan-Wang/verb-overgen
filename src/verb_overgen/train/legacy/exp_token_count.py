from os import environ
import polars as pl
import transformers as hf
from apfel.container.once import Lazy

# Load the GPT 2 tokenizer
TOKENIZER = Lazy(lambda: hf.AutoTokenizer.from_pretrained("gpt2"))

DATASET = pl.read_parquet(
    f"{environ['DATA_PATH_BASE']}/babylm/20250323T140740.aochildes_parsed.parquet"
)

# DATASET = DATASET.filter(pl.col("parse").is_not_null())

DATASET = DATASET.with_columns(
    tokens=pl.col("text").map_elements(lambda text, tok=TOKENIZER(): tok(text).input_ids, return_dtype=pl.List(pl.UInt32))
)

__import__("IPython").embed(using=False)

# DATASET.group_by("split").agg(pl.col("tokens").list.len().sum())
#     imperative(
#         token_count := .alias("token_count"),
#         token_count.sum().over("split", mapping_strategy="explode"),
#     )
# )