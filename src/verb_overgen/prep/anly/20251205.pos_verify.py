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
    import altair as alt
    return alt, mo, pl


@app.cell
def _(pl):
    # dataset = pl.read_parquet("/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0422/merged.test.parquet/pychildes-morphosyntax.20251024.parquet")
    dataset = pl.read_parquet("/home/tomyxw/Projects/2025/verb-overgen/var/pychildes-morphosyntax.20251024.parquet")
    return (dataset,)


@app.cell
def _(dataset, pl):
    dataset_lemma_pos = dataset.lazy().select(
        pl.col("morph").struct.field("pos"),
        pl.col("morph").struct.field("lemma"),
    ).explode(["pos", "lemma"]).group_by("pos").agg(
        pl.col("lemma").value_counts(),
        pl.len().alias("count"),
    ).with_columns(
        pl.col("lemma").list.len().alias("n_unique_lemma")
    ).sort(["count", "n_unique_lemma"], descending=[True, False]).collect()
    return (dataset_lemma_pos,)


@app.cell
def _(dataset_lemma_pos, mo):
    pos_counts = { pos: (count, n_lemma) for pos, count, n_lemma in zip(dataset_lemma_pos["pos"], dataset_lemma_pos["count"], dataset_lemma_pos["n_unique_lemma"]) }
    dropdown_pos_count = mo.ui.dropdown(
        options={ f"{pos} ({count}, {n_lemma})": pos for pos, (count, n_lemma) in pos_counts.items() },
        label="POS Tag",
    )
    return dropdown_pos_count, pos_counts


@app.cell
def _(pl):
    def _find_lemma_pos(morph_expr: pl.Expr, target_lemma: pl.Expr, target_pos: pl.Expr) -> list[pl.Expr]:
        pos = morph_expr.struct.field("pos")
        lemma = morph_expr.struct.field("lemma")

        # is_lemma = lemma.list.eval(lemma.element().eq(lemma).arg_true())
        is_lemma = lemma.list.eval(pl.element().eq(target_lemma).arg_true())
        is_pos = pos.list.eval(pl.element().eq(target_pos).arg_true())

        return [
            lemma.list.contains(target_lemma),
            pos.list.contains(target_pos),
            is_lemma.list.set_intersection(is_pos).list.len() > 0,
        ]

    def find_lemma_pos(df: pl.DataFrame, morph: pl.Expr | None = None, *, lemma: pl.Expr, pos: pl.Expr) -> pl.DataFrame:
        morph = morph if morph is not None else pl.col("morph")
        pre_lemma, pre_pos, match = _find_lemma_pos(morph, lemma, pos)

        return df.filter(pre_lemma, pre_pos, match)
    return (find_lemma_pos,)


@app.cell
def _(mo):
    box_lemma = mo.ui.text(label="Lemma", value="stay")
    return (box_lemma,)


@app.cell
def _(box_lemma, dataset_lemma_pos, mo, pl):
    _available_pos = dataset_lemma_pos.filter(
        pl.col("lemma").list.eval(pl.element().struct.field("lemma").eq(box_lemma.value).arg_true()).list.len() > 0
    )["pos"].to_list()

    box_pos = mo.ui.dropdown(
        label="POS",
        options=_available_pos,
        value=_available_pos[0] if len(_available_pos) > 0 else None,
    )
    return (box_pos,)


@app.cell
def _(dataset_lemma_pos, dropdown_pos_count, mo, pl, pos_counts):
    df_lemma = dataset_lemma_pos.filter(
        pl.col("pos").eq(dropdown_pos_count.value)
    ).select("pos", "lemma").explode("lemma").unnest("lemma") if dropdown_pos_count.value is not None else pl.DataFrame()

    mo.vstack([
        dropdown_pos_count,
        f"{dropdown_pos_count.value} - Total Occurrences: {pos_counts[dropdown_pos_count.value][0]}, Unique Lemmas: {pos_counts[dropdown_pos_count.value][1]}" if dropdown_pos_count.value is not None else "Select a POS tag to see details.",
        df_lemma,
    ])
    return

@app.cell
def _(box_lemma, box_pos, dataset, find_lemma_pos, mo, pl):
    _df = dataset.pipe( 
        lambda df: find_lemma_pos(df, lemma=pl.lit(box_lemma.value), pos=pl.lit(box_pos.value))
    ).select(
        pl.col("id_file"),
        pl.col("id_utterance"),
        pl.col("speaker"),
        pl.col("content"),
        pl.col("morph").struct.field("lemma"),
        pl.col("morph").struct.field("pos"),
        pl.col("morph"),
        pl.col("has_parse"),
        pl.col("syntax"),
    ) if box_pos.value is not None else pl.DataFrame()

    mo.vstack([
        box_lemma,
        box_pos,
        _df,
    ])
    return


if __name__ == "__main__":
    app.run()
