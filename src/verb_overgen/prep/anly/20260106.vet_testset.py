# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    dataset = pl.read_parquet("/home/tomyxw/Projects/2025/verb-overgen/verb-overgen/data/testset/testset_llm_gen.20260108T2325.parquet")
    dataset = dataset.select(
        pl.col("verb"),
        pl.all().exclude("verb", "fragment_object", "fragment_noun_bank"),
    )
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset, pl):
    import regex as re

    expr_suffix = pl.when(
        pl.col("verb").str.contains("o$|ch$|sh$|x$|s$|z$"),
    ).then(pl.lit("es")).otherwise(pl.lit("s")).alias("suffix")

    dataset_output = dataset.select(
        pl.col("verb"),
        pl.col("fragment_subject", "prompt", "llm_gen"),
        pl.col("llm_gen").struct.field("output_text").str.split("\n")
    ).select(
        pl.col("verb"),
        pl.col("fragment_subject", "prompt", "llm_gen"),
        pl.col("output_text").list.first().str.strip_chars().alias("sentence_context"),
        pl.col("output_text").list.get(1).str.strip_chars().alias("sentence_target"),
    ).with_columns(
        pl.struct("sentence_target", "verb", expr_suffix).map_elements(lambda s: (
            re.sub(fr"makes (?P<object>.*) (?P<verb>{s['verb']})", fr"\g<verb>{s['suffix']} \g<object>", s["sentence_target"])
        )).alias("sentence_target_alt"),
        pl.col("sentence_target").str.extract(
        pl.concat_str(
            pl.lit("makes (.*) "),
            pl.col("verb"),
        )
    ).alias("fragment_object"),
    ).group_by("verb").agg(
        pl.struct(pl.all()).sample(3, seed=42).alias("struct")
    ).sort("verb").explode("struct").unnest("struct")

    # dataset_output.write_parquet("/home/tomyxw/Projects/2025/verb-overgen/verb-overgen/data/testset/testset_llm_gen_vetted.20260109V1.parquet")
    return (dataset_output,)


@app.cell
def _(dataset_output, pl):
    dataset_output.with_row_index().unpivot(
        on=pl.selectors.starts_with("sentence_"),
        index="index",
        variable_name="kind",
        value_name="sentence",
    )
    return


if __name__ == "__main__":
    app.run()
