# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.19.2"
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
    import os as _os

    _os.chdir(_os.environ["PWD"])
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

    # LIST_PERIPHRASTIC_CAUS_VERBS.extend([
    #     "cry",
    #     "walk",
    #     "yell",
    #     "run",
    #     "stand",
    #     "gallop",
    #     "cross",
    # ])
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
        pl.col("morph").struct.field("lemma").list.contains(item="make")
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
    return dataset_periphrastic_causative, extract_periphrastic_make


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
def _(
    LIST_PERIPHRASTIC_CAUS_VERBS,
    dataset,
    expr_content_lemma,
    expr_content_pos,
    expr_content_raw,
    extract_periphrastic_make,
    pl,
):
    dataset_all_periphrastic = dataset.filter(
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
        pl.col("verb_causative").is_in(LIST_PERIPHRASTIC_CAUS_VERBS).not_(),
    )
    return (dataset_all_periphrastic,)


@app.cell
def _(dataset_all_periphrastic, pl):
    dataset_all_periphrastic.select(pl.col("verb_causative").value_counts().struct.unnest()).sort(["count", "verb_causative"], descending=[True, False])
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


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset, expr_content_lemma, expr_content_pos, expr_content_raw, pl):
    from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_causative, extract_verb, get_substructure, get_target_word

    dataset_otherwise = dataset.filter(
        pl.col("morph").struct.field("lemma").list.contains(item="get")
    ).select(
        "id_file",
        "id_utterance",
        expr_content_raw,
        expr_content_lemma,
        (_expr_pos := pl.col("morph").struct.field("pos")).list.len().add(2).alias("count_tokens"),
        expr_content_pos,
        # extract_verb(_expr_pos, pl.col("morph").struct.field("lemma")).alias("verb"),
        pl.col("parse").pipe(extract_periphrastic_causative, support_verb="get", pos=_expr_pos).list.first().alias("verb_causative"),
        get_substructure(
            pl.col("parse"),
            pl.col("parse").pipe(get_target_word, target_word="get").list.first(),
            _expr_pos,
        ),
        "morph",
        "parse",
    ).filter(
        pl.col("verb_causative").is_not_null(),
        pl.col("substructure").struct["children_rel"].list.contains("OBJ"),
        # pl.col("verb").list.len() > 0,
        # pl.col("verb_causative").
        # pl.col("verb").list.contains("get"),
        # pl.col("verb").list.eval(pl.element().eq("get").arg_true()).list.first() < (pl.col("verb").list.len() - 1),
        # pl.col("morph").struct.field("lemma").list.contains("to"),
    ).with_row_index()
    return (dataset_otherwise,)


@app.cell
def _(dataset_otherwise, pl):
    dataset_otherwise.group_by("verb_causative").agg(pl.len().alias("count")).sort("count", descending=True)
    return


@app.cell
def _(dataset_otherwise, mo):
    num_example_idx = mo.ui.number(start=0, stop=len(dataset_otherwise) - 1)
    return (num_example_idx,)


@app.cell
def _(dataset_otherwise, mo, num_example_idx):
    from verb_overgen.prep.lib.graph import create_parse_graph, draw_parse_graph

    _fig = draw_parse_graph(
        create_parse_graph(dataset_otherwise[num_example_idx.value, "parse"]), # type: ignore
        highlight_node=lambda g, n, d: ("bold", "red") if d["label"] == "get" else (None, None),
        highlight_edge=lambda g, u, v, d: (
            ("bold", "red") if g.nodes[u]["label"] == "get"
            else ("bold", "blue") if g.nodes[v]["label"] == "get"
            else (None, None)
        ),
    )

    mo.vstack([
        num_example_idx,
        dataset_otherwise[num_example_idx.value, :].to_dicts()[0],
        dataset_otherwise[num_example_idx.value, "content_raw"],
        _fig,
    ])
    return


if __name__ == "__main__":
    app.run()
