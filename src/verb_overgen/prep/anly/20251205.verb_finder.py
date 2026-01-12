# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(
    width="medium",
    css_file="/home/tomyxw/Projects/2025/verb-overgen/verb-overgen/notebook.css",
)


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import matplotlib.pyplot as plt
    import seaborn as sns
    return alt, mo, pl, plt


@app.cell
def _(plt):
    plt.rcParams["font.family"] = "Iosevka"
    return


@app.cell
def _(pl):
    # dataset = pl.read_parquet("/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0422/merged.test.parquet/pychildes-morphosyntax.20251024.parquet")
    dataset = pl.read_parquet("/home/tomyxw/Projects/2025/verb-overgen/data/pychildes-morphosyntax.20251024.parquet")

    dataset = dataset.with_columns(
        pl.struct(
            pl.col("syntax").struct.field("to_index").alias("arc"),
            pl.col("syntax").struct.field("rel").alias("rel"),
            pl.col("morph").struct.field("lemma").alias("word"),
        ).alias("parse")
    )
    return (dataset,)


@app.cell
def _():
    # dataset.select(pl.col("morph").struct.field("pos").list.explode().value_counts().struct.unnest()).sort("count", descending=True)
    return


@app.cell
def _(dataset, pl):
    dataset_lemma_pos = dataset.lazy().select(
        "id_file",
        "id_utterance",
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
def _(dataset_lemma_pos, pl):
    df_verblike_lemma = dataset_lemma_pos.select("pos", "lemma").explode("lemma").unnest("lemma").group_by("lemma", pl.col("pos").is_in(["v", "part",]).alias("is_verb")).agg(
        pl.col("pos"),
        pl.col("count"),
    ).group_by("lemma").agg(
        pl.col("is_verb"),
        pl.col("is_verb").any().alias("has_verb"),
        pl.col("pos"),
        pl.col("count"),
    ).filter(
        pl.col("has_verb")
    ).select(
        pl.all().exclude("has_verb")
    ).explode("is_verb", "pos", "count").with_columns(
        pl.col("count").list.sum().alias("total_count")
    ).pivot(
        index="lemma",
        on="is_verb",
        values=["total_count"],
    ).select(
        "lemma",
        pl.col("true").alias("verb_like_count"),
        pl.col("false").alias("non_verb_like_count"),
    ).sort("verb_like_count", descending=True)
    df_verblike_lemma
    return (df_verblike_lemma,)


@app.cell
def _(df_verblike_lemma, mo):
    verb_like_count_threshold = mo.ui.number(start=1000, stop=df_verblike_lemma.max().item(0, column="verb_like_count"), value=10000)
    return (verb_like_count_threshold,)


@app.cell
def _(alt, df_verblike_lemma, mo, pl, verb_like_count_threshold):
    df_abundant = df_verblike_lemma.with_row_index("rank", 1).filter(
        pl.col("verb_like_count") >= verb_like_count_threshold.value
    )

    _df = df_abundant.unpivot(
        on=pl.selectors.ends_with("_count"),
        index=["rank", "lemma"],
        variable_name="type",
        value_name="count",
    ).with_columns(
        pl.when(pl.col("type").str.starts_with("verb_like")).then(pl.lit("verb_like"))
        .otherwise(pl.lit("non_verb_like")).alias("type")
    )

    mo.vstack([
        verb_like_count_threshold,
        alt.Chart(df_abundant).mark_point().encode(
            x=alt.X("rank:Q"),
            y=alt.Y("verb_like_count:Q"),
            tooltip=["lemma", "verb_like_count", "non_verb_like_count", "rank"]
        ),
        alt.Chart(_df).mark_bar().encode(
            x=alt.X("lemma:O", sort=alt.SortField(field="rank", order="ascending")),
            y=alt.Y("count:Q"),
            color=alt.Color("type:N", scale=alt.Scale(domain=["verb_like", "non_verb_like"], range=["#1f77b4", "#ff7f0e"])),
            tooltip=["lemma", "count", "rank"],
        ).configure_axis(
            labelAngle=-45
        ).interactive()
    ])
    return


@app.cell
def _(mo):
    lemma_search = mo.ui.text(label="Lemma", placeholder="laugh", value="laugh")
    return (lemma_search,)


@app.cell
def _(alt, df_verblike_lemma, lemma_search, mo, pl):
    from verb_overgen.prep.lib.lexical import BIDGOOD_VERBS

    def bidgood(word):
        if word in BIDGOOD_VERBS["transitive_only"]:
            return "transitive"
        elif word in BIDGOOD_VERBS["intransitive_only"]:
            return "intransitive"
        elif word in BIDGOOD_VERBS["alternating"]:
            return "alternating"
        else:
            return "uncategorized"


    _df = df_verblike_lemma.with_columns(
        pl.col("lemma").map_elements(bidgood, pl.String).alias("bidgood")
    )

    _chart = alt.Chart(_df).mark_point().encode(
        x=alt.X("non_verb_like_count:Q", title="Non-verb-like Count", scale=alt.Scale(type="log")),
        y=alt.Y("verb_like_count:Q", title="Verb-like Count", scale=alt.Scale(type="log")),
        color=alt.when(lemma=lemma_search.value).then(alt.value("pink")).otherwise(alt.Color("bidgood:N")),
        opacity=alt.when(bidgood="uncategorized").then(alt.value(0.125)).otherwise(alt.value(1.0)),
        tooltip=["lemma:N", "verb_like_count:Q", "non_verb_like_count:Q", "bidgood:N"],
    ).interactive()

    # _chart = alt.Chart(_df).mark_point().encode(
    #     x=alt.X("non_verb_like_count:Q", title="Non-verb-like Count", scale=alt.Scale(type="log")),
    #     y=alt.Y("verb_like_count:Q", title="Verb-like Count", scale=alt.Scale(type="log")),
    #     color=alt.when(lemma=lemma_search.value).then(alt.value("red")).otherwise(alt.value("steelblue")),
    #     tooltip=["lemma:N", "verb_like_count:Q", "non_verb_like_count:Q"],
    # ).interactive()

    mo.vstack([
        lemma_search,
        _chart,
    ])
    return (bidgood,)


@app.cell
def _(bidgood, df_verblike_lemma, pl):
    _df = df_verblike_lemma.with_columns(
        pl.col("lemma").map_elements(bidgood, pl.String).alias("bidgood")
    )

    _df.filter(
        (pl.col("verb_like_count") > 1000).or_(pl.col("bidgood").ne("uncategorized"))
    ).write_csv("verb_candidate.csv")
    return


@app.cell
def _(dataset, lemma_search, pl):
    df_this_lemma = dataset.filter(
        pl.col("morph").struct.field("lemma").list.contains(lemma_search.value)
    )

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
    return (
        df_this_lemma,
        expr_content_lemma,
        expr_content_pos,
        expr_content_raw,
    )


@app.cell
def _():
    from verb_overgen.prep.lib.graph import create_parse_graph, draw_parse_graph
    from verb_overgen.prep.lib.pattern_matching import get_substructure, get_target_word
    return (
        create_parse_graph,
        draw_parse_graph,
        get_substructure,
        get_target_word,
    )


@app.cell
def _(
    df_this_lemma,
    expr_content_lemma,
    expr_content_pos,
    expr_content_raw,
    get_substructure,
    get_target_word,
    lemma_search,
    pl,
):
    df_this_lemma_with_parse = df_this_lemma.with_row_index("index").with_columns(
        get_target_word(pl.col("parse"), lemma_search.value)
    ).explode("index_target_word").select(
        # "id_file",
        # "id_utterance",
        "index",
        expr_content_raw, expr_content_lemma, expr_content_pos,
        get_substructure(pl.col("parse"), pl.col("index_target_word"), pl.col("morph").struct.field("pos")).struct.unnest(),
        "parse",
    )

    df_this_lemma_with_parse
    return (df_this_lemma_with_parse,)


@app.cell
def _(df_this_lemma_with_parse, pl):
    df_this_lemma_with_parse.filter(pl.col("children_rel").list.contains("OBJ"))
    return


@app.cell
def _(df_this_lemma_with_parse, lemma_search, mo, pl):
    mo.vstack([
        mo.md(f"### Children Statistics of *{lemma_search.value}*"),
        (
            df_this_lemma_with_parse
                .explode(pl.selectors.starts_with("children"))
                .group_by("children_rel")
                .agg(
                    pl.count()
                )
                .sort("count", descending=True)
        ),
        (
            df_this_lemma_with_parse
                .select(pl.col("children_rel").value_counts().struct.unnest())
                .sort("count", descending=True)
        )
    ])
    return


@app.cell
def _(df_this_lemma_with_parse, mo):
    num_example_idx = mo.ui.number(start=0, stop=len(df_this_lemma_with_parse) - 1)
    return (num_example_idx,)


@app.cell
def _(
    create_parse_graph,
    df_this_lemma_with_parse,
    draw_parse_graph,
    lemma_search,
    mo,
    num_example_idx,
):
    _fig = draw_parse_graph(
        create_parse_graph(df_this_lemma_with_parse[num_example_idx.value, "parse"]),
        highlight_node=lambda g, n, d: ("bold", "red") if d["label"] == lemma_search.value else (None, None),
        highlight_edge=lambda g, u, v, d: (
            ("bold", "red") if g.nodes[u]["label"] == lemma_search.value
            else ("bold", "blue") if g.nodes[v]["label"] == lemma_search.value
            else (None, None)
        ),
    )

    mo.vstack([
        num_example_idx,
        df_this_lemma_with_parse[num_example_idx.value, "content_raw"],
        _fig,
    ])
    return


@app.cell
def _(df_this_lemma_with_parse):
    df_this_lemma_with_parse
    return


@app.cell
def _(mo):
    mo.md(r"""
    break: alternating

    laugh: intransitive


    train: The baby laughed
    vs Jessy made the baby laugh

    test: John laughed Mary
    vs John made Mary laugh
    """)
    return


@app.cell
def _(df_this_lemma_with_parse, pl):


    (
        df_this_lemma_with_parse
            .filter(
                pl.col("head").is_in(["make", "let", "get"]),
                # pl.col("head")
            )
    )
    return


@app.cell
def _():
    from verb_overgen.prep.lib.lexical import BIDGOOD_VERBS

    BIDGOOD_VERBS
    return


if __name__ == "__main__":
    app.run()
