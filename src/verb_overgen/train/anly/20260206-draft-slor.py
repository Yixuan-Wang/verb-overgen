import polars as pl

from verb_overgen.train.train import TrainingSession
from verb_overgen.train.lib.utils import load_train_config


cfg = load_train_config()
session = TrainingSession(cfg)

tokenizer = session.setup_tokenizer()
datasets = session.setup_datasets()
dataset_train: pl.DataFrame = datasets["train"].dataframe # type: ignore

dataset_unigram_count = dataset_train.lazy().select("input_ids").explode("input_ids").select(pl.col("input_ids").value_counts().struct.unnest()).sort("input_ids").collect()


df = dataset_train.head(5)

def get_ngrams(input_ids: pl.Expr, n: int) -> list[pl.Expr]:
    ngrams = [
        input_ids.alias("token-1")
    ]
    
    for i in range(1, n):
        ngrams.append(input_ids.list.shift(-i).alias(f"token-{i+1}"))
        
    return ngrams

def get_ngram_count(dataset: pl.DataFrame, n: int) -> pl.DataFrame:
    df: pl.DataFrame = ( # type: ignore
        dataset
        .lazy()
        .select(
            get_ngrams(pl.col("input_ids"), n)
        )
        .explode(pl.selectors.starts_with("token-"))
        .drop_nulls()
        .group_by(pl.selectors.starts_with("token-"))
        .len()
        .sort("len", descending=True)
        .select(
            pl.concat_list(pl.selectors.starts_with("token-")).alias("ngram"),
            pl.col("len").alias("count")
        )
        .collect()
    )
    
    return df

dataset_bigram_count = get_ngram_count(dataset_train, 2)
dataset_trigram_count = get_ngram_count(dataset_train, 3)
dataset_pentagram_count = get_ngram_count(dataset_train, 5)

dataset_unigram_count.write_parquet("./var/pychildes/ngram/token_count.parquet")

dataset_ngram_count = pl.concat([
    dataset_unigram_count.select(
        pl.concat_list("input_ids").alias("ngram"),
        "count",
    ),
    dataset_bigram_count,
    dataset_trigram_count,
    dataset_pentagram_count,    
])

dataset_ngram_count = dataset_ngram_count.with_columns(
    pl.col("ngram").list.len().alias("n")
)

dataset_ngram_count.write_parquet("./var/pychildes/ngram/ngram_count.parquet")
