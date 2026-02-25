# /// script
# [tool.marimo]
# runtime.autoreload = "autorun"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from functools import partial

    import polars as pl

    from verb_overgen.prep.lib.graph import create_parse_graph, draw_parse_graph
    from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_verb
    from verb_overgen.prep.lib.lexical import LIST_CAUSATIVE_VERBS

    return (
        LIST_CAUSATIVE_VERBS,
        create_parse_graph,
        draw_parse_graph,
        extract_periphrastic_verb,
        os,
        partial,
        pl,
    )


@app.cell
def _(os):
    os.chdir(os.environ["PWD"])
    return


@app.cell
def _(pl):
    dataset = pl.read_parquet("./data/pychildes/pychildes-morphosyntax_postprocess.20260212_20251024.parquet")
    return (dataset,)


@app.cell
def _(create_parse_graph, draw_parse_graph, pl):
    def get_entry(df, id_file: str, id_utterance: int, key: str | None = None):
        _df = df.filter(
            pl.col("id_file").str.contains(id_file),
            pl.col("id_utterance").eq(id_utterance),
        )
        if len(_df) == 0:
            raise KeyError("No entry found for id_file={} and id_utterance={}".format(id_file, id_utterance))
        elif len(_df) > 1:
            raise KeyError("Multiple entries found for id_file={} and id_utterance={}".format(id_file, id_utterance))
        return _df[0].item(0, key) if key is not None else _df[0]


    def draw(parse, verb):
        return draw_parse_graph(
            create_parse_graph(parse),
            highlight_node=lambda g, n, d: ("bold", "red") if d["label"] == verb else (None, None),
            highlight_edge=lambda g, u, v, d: (
                ("bold", "red")
                if g.nodes[u]["label"] == verb
                else ("bold", "blue")
                if g.nodes[v]["label"] == verb
                else (None, None)
            ),
        )

    return draw, get_entry


@app.cell
def _(dataset, pl):
    df_make = dataset.filter(pl.col("morph").struct.field("lemma").list.contains(item="make"))
    return


@app.cell
def _(dataset, extract_periphrastic_verb, partial, pl):
    dataset.select(
        pl.col("parse").map_elements(partial(extract_periphrastic_verb, support_verb="make"), pl.List(pl.String))
    )
    return


@app.cell
def _(dataset, partial, pl):
    df = dataset.filter(pl.col("morph").struct.field("lemma").list.contains("let"))
    df_let = df.with_columns(
        pl.col("featured_verbs").list.eval(pl.element().str.strip_prefix("causative:")),
        pl.col("parse")
        .map_elements(partial(_extract_periphrastic_verb, support_verb="let"), pl.List(pl.String))
        .alias("filtered_verbs"),
    )
    return (df_let,)


@app.cell
def _():
    from verb_overgen.train.lib.validate import VERB_TARGET_5

    VERB_TARGET_5
    return


@app.cell
def _(LIST_CAUSATIVE_VERBS, df_let, pl):
    df_let_1 = df_let.filter(
        pl.col("filtered_verbs").list.len() > 0,
        pl.col("filtered_verbs").list.set_intersection(LIST_CAUSATIVE_VERBS).list.len() == pl.col("filtered_verbs").list.len(),
    )

    df_let_1
    return (df_let_1,)


@app.cell
def _(df_let_1, pl):
    df_let_1.select(pl.col("filtered_verbs").list.explode()).group_by("filtered_verbs").agg(pl.len().alias("count")).sort("count", descending=True).select(
        pl.col("filtered_verbs").alias("let"),
        pl.col("count"),
    )
    return


@app.cell
def _(dataset, pl):
    dataset.filter(
        pl.col("features").list.contains("causative"),
    ).select(
        pl.col("featured_verbs").list.eval(pl.element().str.strip_prefix("causative:")).list.explode()
    ).group_by("featured_verbs").agg(pl.len().alias("count")).sort("count", descending=True).select(
        pl.col("featured_verbs").alias("make"),
        pl.col("count"),
    )
    return


@app.cell
def _(dataset, draw, get_entry):
    draw(
        get_entry(dataset, "Eng-UK/Manchester/Joel/020626", 450, "parse"),
        "go",
    )
    return


if __name__ == "__main__":
    app.run()
