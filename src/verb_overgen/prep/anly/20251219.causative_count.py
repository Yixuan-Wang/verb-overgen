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
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.family"] = "Iosevka"
    return mo, pl


@app.cell
def _():
    __import__("os").chdir(__import__("os").environ["VSCODE_CWD"])
    return


@app.cell
def _(pl):
    dataset = pl.read_parquet("./data/pychildes-morphosyntax.20251024.parquet")

    dataset = dataset.with_columns(
        pl.struct(
            pl.col("syntax").struct.field("to_index").alias("arc"),
            pl.col("syntax").struct.field("rel").alias("rel"),
            pl.col("morph").struct.field("lemma").alias("word"),
        ).alias("parse")
    )
    return (dataset,)


@app.cell
def _(dataset, pl):
    dataset_verb_list = dataset.lazy().select(
        "id_file",
        "id_utterance",
        pl.col("morph").struct.field("pos"),
        pl.col("morph").struct.field("lemma"),
    ).explode(["pos", "lemma"]).group_by("lemma").agg(
        pl.col("pos").is_in(["v", "part"]).sum().alias("count_is_verb"),
        pl.len().alias("count"),
    ).filter(pl.col("count_is_verb") > 0).sort("count_is_verb", descending=True).collect()
    return (dataset_verb_list,)


@app.cell
def _():
    LIST_PERIPHRASTIC_CAUS_VERBS = """go
    come
    fall
    talk
    run
    stay
    sleep
    sing
    drink
    jump
    live
    swim
    laugh
    die
    smile
    hop
    appear
    arrive
    tumble
    giggle
    rise
    glow
    leak
    moan
    enter
    grin
    frown
    chuckle
    groan
    vanish
    glitter
    descend
    exit
    chortle""".split("\n")
    return (LIST_PERIPHRASTIC_CAUS_VERBS,)


@app.cell
def _(mo):
    lemma_search = mo.ui.text(label="Lemma Search", value="go")
    return (lemma_search,)


@app.cell
def _(lemma_search, pl):
    expr_content_raw = pl.col("content").alias("content_raw")
    expr_content_lemma = (
            pl.col("morph").struct.field("lemma")
                .list.join(" ")
                .str.replace(r".+", "[$0]")
                .str.replace(fr"({lemma_search.value})", "{$1}")
                .alias("content_lemma")
        )
    expr_content_pos = (
            pl.col("morph").struct.field("pos")
                .list.join(" ")
                .str.replace(r".+", "[$0]")
                .alias("content_pos")
        )
    return expr_content_lemma, expr_content_pos, expr_content_raw


@app.cell
def _(
    LIST_PERIPHRASTIC_CAUS_VERBS,
    dataset,
    expr_content_lemma,
    expr_content_pos,
    expr_content_raw,
    pl,
):
    from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_make

    dataset_periphrastic_causative = dataset.filter(
        pl.col("morph").struct.field("lemma").list.contains("make")
    ).select(
        "id_file",
        "id_utterance",
        expr_content_raw,
        expr_content_lemma,
        pl.col("morph").struct.field("pos").list.len().add(2).alias("count_tokens"),
        expr_content_pos,
        pl.col("parse").pipe(extract_periphrastic_make).list.first().alias("verb_causative"),
    ).filter(
        pl.col("verb_causative").is_not_null(),
        pl.col("verb_causative").is_in(LIST_PERIPHRASTIC_CAUS_VERBS),
    )
    return (dataset_periphrastic_causative,)


@app.cell
def _(
    LIST_PERIPHRASTIC_CAUS_VERBS,
    dataset_periphrastic_causative,
    dataset_verb_list,
    pl,
):
    dataset_verb_list.filter(
        pl.col("lemma").is_in(LIST_PERIPHRASTIC_CAUS_VERBS)
    ).sort("count", descending=True).join(
        dataset_periphrastic_causative.group_by("verb_causative").agg(
            pl.len().alias("count_periphrastic_causative")
        ),
        left_on="lemma",
        right_on="verb_causative",
        how="left"
    ).sort(["count_is_verb", "count_periphrastic_causative"], descending=True)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(dataset_periphrastic_causative, lemma_search, mo, pl):
    _df = dataset_periphrastic_causative.filter(pl.col("verb_causative") == lemma_search.value)

    mo.vstack([
        lemma_search,
        _df.select(pl.col("count_tokens").sum()),
        _df,
    ])
    return


if __name__ == "__main__":
    app.run()
