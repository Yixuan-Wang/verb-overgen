"""Add Spacy parses to SCAMP dataset.

Reasons behind this addition:
- Provide alignment information between paired SCAMP sentences.
- Besides, our training uses lemmatized sentences to reduce vocabulary size and improve token reuse.
"""

import polars as pl
import pyarrow as pa
from loguru import logger
import ray
import spacy
from spacy.tokens import Doc

class SpacyParser:
    def __init__(self):
        self.spacy_pipeline = spacy.load("en_core_web_trf", disable=["ner"])

    def run_spacy(self, words: list[list[str]]) -> pa.Array:
        inputs = [Doc(self.spacy_pipeline.vocab, words=w) for w in words]
        outputs = list(self.spacy_pipeline.pipe(inputs))

        lemma = pa.array([[token.lemma_ for token in doc] for doc in outputs], pa.list_(pa.string()))
        morph = pa.array([[str(token.morph) for token in doc] for doc in outputs], pa.list_(pa.string()))
        dep_arc = pa.array(
            [[(token.head.i + 1 if token.dep_ != "ROOT" else 0) for token in doc] for doc in outputs],
            pa.list_(pa.int32()),
        )
        dep_rel = pa.array([[token.dep_ for token in doc] for doc in outputs], pa.list_(pa.string()))
        pos_ud = pa.array([[token.pos_ for token in doc] for doc in outputs], pa.list_(pa.string()))
        pos_penn = pa.array([[token.tag_ for token in doc] for doc in outputs], pa.list_(pa.string()))

        return pa.StructArray.from_arrays(
            [lemma, morph, dep_arc, dep_rel, pos_ud, pos_penn],
            names=["lemma", "morph", "dep_arc", "dep_rel", "pos_ud", "pos_penn"],
        )

    def __call__(self, table: pa.Table) -> pa.Table:
        list_word = table.column("list_word")
        arr_roberta = self.run_spacy(list_word.to_pylist())
        return table.append_column("roberta", arr_roberta)


if __name__ == "__main__":
    dataset = pl.read_parquet("./data/scamp/scamp-plausible.parquet")

    dataset = dataset.with_columns(
        pl.col("sentence_raw").str.split(" ").alias("list_word"),
    )

    ray.init(num_cpus=48)
    dataset = ray.data.from_arrow(dataset.to_arrow())

    dataset = dataset.repartition(47).map_batches(
        SpacyParser, # pyright: ignore
        batch_size=1024,
        batch_format="pyarrow",
        compute=ray.data.ActorPoolStrategy(min_size=16, max_size=47),
        num_cpus=1,
        scheduling_strategy="SPREAD",
    )
    dataset.write_parquet("./data/scamp/scamp-plausible-roberta.parquet")
