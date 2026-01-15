import polars as pl

from verb_overgen.prep.lib.lexical import BIDGOOD_VERBS

from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_make

def load_data(file_path: str) -> pl.DataFrame:
    """Load data.

    Extra steps:
    - Create a `parse` column following the `[arc, rel, word]` format, if not present.
    """
    dataset = pl.read_parquet(file_path)

    if "parse" not in dataset.columns:
        dataset = dataset.with_columns(
            pl.struct(
                pl.col("syntax").struct.field("to_index").alias("arc"),
                pl.col("syntax").struct.field("rel").alias("rel"),
                pl.col("morph").struct.field("lemma").alias("word"),
            ).alias("parse")
        )

    return dataset

LIST_CAUSATIVE_VERBS = list(BIDGOOD_VERBS["intransitive_only"])

def count_verb(dataset: pl.DataFrame):
    dataset_verb_list = (
        dataset.select(
            "id_file",
            "id_utterance",
            pl.col("morph").struct.field("pos"),
            pl.col("morph").struct.field("lemma"),
        )
        .explode(["pos", "lemma"])
        .group_by("lemma")
        .agg(
            pl.col("pos").is_in(["v", "part"]).sum().alias("count_is_verb"),
            pl.len().alias("count"),
        )
        .filter(pl.col("count_is_verb") > 0)
        .sort("count_is_verb", descending=True)
    )

    return dataset_verb_list

def count_top_noun(dataset: pl.DataFrame, lower_bound: int = 5000):
    dataset_noun_list = (
        dataset.select(
            "id_file",
            "id_utterance",
            pl.col("morph").struct.field("pos"),
            pl.col("morph").struct.field("lemma"),
        )
        .explode(["pos", "lemma"])
        .group_by("lemma")
        .agg(
            pl.col("pos").eq("n").sum().alias("count_is_noun"),
            pl.len().alias("count"),
        )
        .filter(pl.col("count_is_noun") > lower_bound)
        .sort("count_is_noun", descending=True)
    )

    return dataset_noun_list

if __name__ == "__main__":
    df = load_data("./data/pychildes-morphosyntax.20251024.parquet")

    # Verb counts
    if False:
        verb_counts = count_verb(df)
        df_candidate_count = verb_counts.filter(pl.col("lemma").is_in(LIST_CAUSATIVE_VERBS))

        df_causative_count = df.select(
            pl.col("parse").pipe(extract_periphrastic_make).list.first().alias("verb_causative")
        ).filter(
            pl.col("verb_causative").is_not_null(),
            pl.col("verb_causative").is_in(LIST_CAUSATIVE_VERBS),
        ).group_by("verb_causative").agg(
            pl.len().alias("count_causative")
        )

        df_candidate_count = df_candidate_count.join(
            df_causative_count,
            left_on="lemma",
            right_on="verb_causative",
            how="left",
        ).sort(["count_is_verb", "count_causative"], descending=True).fill_null(0)

        df_candidate_count.write_csv("./var/causative_verb_counts.csv")

    # Top noun counts
    if True:
        noun_counts = count_top_noun(df, lower_bound=1000).filter(
            pl.col("lemma").str.contains("^[A-Z]").not_()
        )
        noun_counts.write_csv("./var/top_nouns.csv")
