# /// script
# [tool.marimo]
# runtime.autoreload = "autorun"
# ///

import marimo

__generated_with = "0.20.1"
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
    dataset_main = pl.read_parquet("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")
    dataset_annotation = pl.read_parquet("./data/pychildes/pychildes-morphosyntax_annotation-20260221.20251024.parquet")

    dataset = dataset_main.join(dataset_annotation, on=["id_file", "id_utterance"])
    return (dataset,)


@app.cell
def _(pl):
    expr_parse = pl.struct(
                pl.col("syntax").struct.field("to_index").alias("arc"),
                pl.col("syntax").struct.field("rel").alias("rel"),
                pl.col("morph").struct.field("lemma").alias("word"),
            ).alias("parse")
    return (expr_parse,)


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
    return (df_make,)


@app.cell
def _(df_make, expr_parse, extract_periphrastic_verb, partial, pl):
    df_make_annotated = df_make.with_columns(
        expr_parse
        .map_elements(partial(extract_periphrastic_verb, support_verb="let"), pl.List(pl.String))
        .alias("target_verb"),
    )
    return (df_make_annotated,)


@app.cell
def _():
    from verb_overgen.train.lib.validate import VERB_TARGET_5

    VERB_TARGET_5
    return


@app.cell
def _(df_make_annotated, pl):
    df_make_extended_lemma = df_make_annotated.filter(
        pl.col("features").list.contains("causative").not_(),
        pl.col("target_verb").list.len().gt(0),
    ).select(
        pl.col("target_verb").list.explode().alias("word").value_counts().struct.unnest()
    ).sort("count", descending=True)
    return (df_make_extended_lemma,)


@app.cell
def _(df_make_extended_lemma):
    df_make_extended_lemma
    return


@app.cell
def _(pl):
    df_extended_lemma_annotated = pl.read_csv("./var/pychildes/lexical/extended_lemma.csv").select(pl.all().exclude("count"))
    return (df_extended_lemma_annotated,)


@app.cell
def _(df_extended_lemma_annotated):
    df_extended_lemma_is_verb = df_extended_lemma_annotated.filter("is_usually_verb").select("word")
    # df_extended_lemma_is_verb.write_csv("./data/pychildes/lexical/extended_causative_verb_lemma.csv")

    list_extended_verb = df_extended_lemma_is_verb.to_series().to_list()
    return (list_extended_verb,)


@app.cell
def _(pl):
    pl.read_csv("./data/pychildes/lexical/extended_causative_verb_lemma.csv").select("word").to_series().to_list()
    return


@app.cell
def _(df_make_annotated, list_extended_verb, pl):
    df_make_annotated.filter(
        pl.col("target_verb").list.set_intersection(list_extended_verb).list.len() > 0
    )
    return


@app.cell
def _(df_make_annotated, pl):
    df_make_annotated.filter(pl.col("features").list.contains("causative"))
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
