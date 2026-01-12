"""Merge different SCAMP data into a single table."""

from collections.abc import Iterable

import pyarrow as pa
import polars as pl
import regex as re

from typing import cast

from src.verb_overgen.prep.lib.lexical import BaseSpacyParser

if __name__ != "__main__":
    raise RuntimeError("This script is not intended to be imported as a module.")


def load_dataset(
    input_path: str,
    *,
    sample_n: int = 3,
    sample_seed: int = 42,
) -> pl.DataFrame:
    dataset = pl.read_parquet(input_path)
    dataset = dataset.select(
        pl.col("verb"),
        pl.all().exclude("verb", "fragment_object", "fragment_noun_bank"),
    )

    # expr_suffix = (
    #     pl.when(
    #         pl.col("verb").str.contains("o$|ch$|sh$|x$|s$|z$"),
    #     )
    #     .then(pl.lit("es"))
    #     .otherwise(pl.lit("s"))
    #     .alias("suffix")
    # )

    def alternate_transitive(s):
        sentence_target, verb = s["sentence_target"], s["verb"]
        if re.search("o$|ch$|sh$|x$|s$|z$", verb):
            return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", r"\g<verb>es \g<object>", sentence_target)
        elif re.search("y$", verb) and not re.search("[aeiou]y$", verb):
            return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", rf"{verb[:-1]}ies \g<object>", sentence_target)
        else:
            return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", r"\g<verb>s \g<object>", sentence_target)

    dataset_output = (
        dataset.select(
            pl.col("verb"),
            pl.col("fragment_subject", "prompt", "llm_gen"),
            pl.col("llm_gen").struct.field("output_text").str.split("\n"),
        )
        .select(
            pl.col("verb"),
            pl.col("fragment_subject", "prompt", "llm_gen"),
            pl.col("output_text").list.first().str.strip_chars().alias("sentence_context"),
            pl.col("output_text").list.get(1).str.strip_chars().alias("sentence_target"),
        )
        .with_columns(
            pl.struct("sentence_target", "verb")
            .map_elements(alternate_transitive)
            .alias("sentence_target_alt"),
            pl.col("sentence_target")
            .str.extract(
                pl.concat_str(
                    pl.lit("makes (.*) "),
                    pl.col("verb"),
                )
            )
            .alias("fragment_object"),
        )
        # .group_by("verb")
        # .agg(pl.struct(pl.all()).sample(sample_n, seed=sample_seed).alias("struct"))
        .sort("verb")
        # .explode("struct")
        # .unnest("struct")
        .with_row_index()
    )

    return dataset_output


class SpacyParser(BaseSpacyParser):
    """A Ray data batch mapper that adds Spacy parses to testset dataset."""

    def __call__(self, table: pa.Table) -> pa.Table:
        sentence = table.column("sentence")
        arr_roberta = self.run_spacy(sentences=sentence.to_pylist())
        return table.append_column("roberta", arr_roberta)


def unpivot_dataset(df: pl.DataFrame) -> pl.DataFrame:
    return df.unpivot(
        on=pl.selectors.starts_with("sentence_"),
        index="index",
        variable_name="kind",
        value_name="sentence",
    )


def pivot_dataset(
    df_roberta: pl.DataFrame,
) -> pl.DataFrame:
    return df_roberta.pivot(on="kind", index="index", values=["sentence", "roberta"]).select(
        pl.all().name.replace("_sentence_", "_")
    )


if __name__ == "__main__":
    dataset = load_dataset("data/testset/output/testset_llm_gen_supplementary.20260108T2325.parquet")
    dataset_long = unpivot_dataset(dataset)

    parser = SpacyParser()
    dataset_roberta = cast(pl.DataFrame, pl.from_arrow(parser(dataset_long.to_arrow())))
    dataset_roberta_wide = pivot_dataset(dataset_roberta)

    dataset = dataset.join(dataset_roberta_wide, on="index", coalesce=True)

    dataset.write_parquet("./data/testset/output/testset_llm_gen_supplementary-postprocessed.20260108T2325.parquet")
