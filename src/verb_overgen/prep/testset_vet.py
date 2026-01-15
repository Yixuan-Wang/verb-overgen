"""
Process the manually vetted testset data to merge with the original one.
"""

import polars as pl
from collections.abc import Sequence


def load_data_vetted(input_path: str) -> pl.DataFrame:
    """Load the vetted testset data from a CSV file."""
    df = (
        pl.read_csv(
            input_path,
            schema_overrides={
                "Kanishka": pl.Int64,
                "Tom": pl.Int64,
            },
        )
        .rename(
            {
                "Kanishka": "vet_verdict_kanishka",
                "Tom": "vet_verdict_tom",
                "Notes": "vet_note_kanishka",
                "Tom Notes": "vet_note_tom",
            }
        )
        .select(
            pl.col("index"),
            pl.col("verb"),
            pl.struct(pl.selectors.starts_with("vet").name.replace("^vet_", "")).alias("vet"),
        )
    )
    return df


def load_data_original(input_path: str) -> pl.DataFrame:
    """Load the original testset data from a Parquet file."""
    df = pl.read_parquet(input_path)
    df = df.select(
        pl.col("index"),
        pl.col("verb"),
        pl.struct(pl.col("llm_gen").struct.field("output_text")),
        pl.col("llm_gen").struct.with_fields(pl.col("fragment_subject"), pl.col("prompt")),
    )
    return df


def vet(
    dataset_vetted: pl.DataFrame,
    dataset_original: pl.DataFrame,
    # target_verbs: Sequence[str] = ("laugh", "cry", "sing", "smile", "sleep"),
):
    """
    Append the vetted data to the original dataset.
    """
    dataset = dataset_original.join(
        # keep verb here to verify correctness later
        dataset_vetted,
        on=["index"],
        how="left",
        coalesce=True,
    )

    assert dataset.select(pl.col("verb").eq(pl.col("verb_right")).all()).item()
    dataset = dataset.drop("verb_right")
    

    # dataset = dataset.filter(
    #     pl.col("verb").is_in(target_verbs),
    #     # pl.col("vet_verdict_kanishka").is_null().or_(pl.col("vet_verdict_kanishka") == 1).and_(
    #     #     pl.col("vet_verdict_tom").is_null().or_(pl.col("vet_verdict_tom") == 1),
    #     # )
    # )

    return dataset
