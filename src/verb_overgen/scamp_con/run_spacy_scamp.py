"""Add Spacy parses to SCAMP dataset.

Reasons behind this addition:
- Provide alignment information between paired SCAMP sentences.
- Besides, our training uses lemmatized sentences to reduce vocabulary size and improve token reuse.
"""

import pyarrow as pa
import polars as pl
import ray

from src.verb_overgen.prep.lib.lexical import BaseSpacyParser

class SpacyParser(BaseSpacyParser):
    """A Ray data batch mapper that adds Spacy parses to SCAMP dataset."""

    def __call__(self, table: pa.Table) -> pa.Table:
        list_word = table.column("list_word")
        arr_roberta = self.run_spacy(words=list_word.to_pylist())
        return table.append_column("roberta", arr_roberta)

if __name__ == "__main__":
    dataset = pl.read_parquet("./data/scamp/scamp-plausible.parquet")

    dataset = dataset.with_columns(
        pl.col("sentence_raw").str.split(" ").alias("list_word"),
    )

    ray.init(num_cpus=48)
    dataset = ray.data.from_arrow(dataset.to_arrow())

    dataset = dataset.repartition(47).map_batches(
        SpacyParser,  # pyright: ignore
        batch_size=1024,
        batch_format="pyarrow",
        compute=ray.data.ActorPoolStrategy(min_size=16, max_size=47),
        num_cpus=1,
        scheduling_strategy="SPREAD",
    )
    dataset.write_parquet("./data/scamp/scamp-plausible-roberta.parquet")
