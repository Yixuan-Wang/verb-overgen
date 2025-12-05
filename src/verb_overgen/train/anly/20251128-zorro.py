import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    import altair as alt
    return alt, mo, pl


@app.cell
def _(pl):
    # dataset = pl.read_parquet("/u501/y4466wan/projects/2025/verb-overgen/data/zorro/eval/zorro-eval_silver-salad-4_final.parquet").unnest("prob")
    dataset = pl.read_parquet("/u501/y4466wan/projects/2025/verb-overgen/data/scamp/eval/scamp-eval_silver-salad-4_final.parquet").unnest("prob")
    return (dataset,)


@app.cell
def _(dataset):
    dataset.head(10)
    return


@app.cell
def _(alt, dataset, pl):
    df_accuracy = dataset.group_by("file_name").agg(
        pl.col("prob_grammatical").gt(pl.col("prob_ungrammatical")).mean().alias("accuracy"),
        pl.count().alias("num_samples"),
    ).sort("accuracy", descending=True)

    alt.Chart(df_accuracy).mark_bar().encode(
        x=alt.X("accuracy:Q"),
        y=alt.Y("file_name:O", sort="-x"),
        color=alt.Color("accuracy:Q", scale=alt.Scale(scheme="greens")),
        tooltip=[alt.Tooltip("file_name:O"), alt.Tooltip("accuracy:Q").format(".2%")],
    ).properties(
        title="Distribution of Zorro Accuracy over Files"
    )
    return (df_accuracy,)


@app.cell
def _(df_accuracy, mo):
    all_file_name = {
        f"{file_name} ({acc:.2%} of {num_samples})": file_name
        for file_name, acc, num_samples
        in zip(df_accuracy["file_name"], df_accuracy["accuracy"], df_accuracy["num_samples"])
    }
    file_name = mo.ui.dropdown(all_file_name, value=next(iter(all_file_name.keys())))
    return (file_name,)


@app.cell
def _(dataset, file_name, mo, pl):
    df_file = dataset.filter(pl.col("file_name") == file_name.value).select(
        pl.col("sentence_grammatical"),
        pl.col("sentence_raw_grammatical"),
        pl.col("sentence_ungrammatical"),
        pl.col("sentence_raw_ungrammatical"),
        pl.col("prob_grammatical").map_elements(lambda x: f"{x:.4e}").alias("prob_grammatical"),
        pl.col("prob_ungrammatical").map_elements(lambda x: f"{x:.4e}").alias("prob_ungrammatical"),
        pl.col("prob_grammatical").gt(pl.col("prob_ungrammatical")).alias("judge"),
    )

    filter_wrong = mo.ui.checkbox(label="Show only wrong judgements", value=False)
    return df_file, filter_wrong


@app.cell
def _(df_file, file_name, filter_wrong, mo, pl):
    if filter_wrong.value:
        df_file_show = df_file.filter(pl.col("judge").eq(False))
    else:
        df_file_show = df_file

    mo.vstack([
        file_name,
        filter_wrong,
        df_file_show,
    ])
    return


if __name__ == "__main__":
    app.run()
