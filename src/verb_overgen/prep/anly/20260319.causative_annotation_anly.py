# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import polars as pl

    os.chdir(os.environ["PWD"])
    return mo, pl


@app.cell
def _():
    from verb_overgen.common.features import VERB_TARGET_5, VERB_TARGET_17

    return VERB_TARGET_17, VERB_TARGET_5


@app.cell
def _():
    from verb_overgen.prep.lib.graph import create_parse_graph, draw_parse_graph

    return create_parse_graph, draw_parse_graph


@app.cell
def _(pl):
    dataset = pl.read_parquet(
        "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"
    )
    return (dataset,)


@app.cell
def _(create_parse_graph, draw_parse_graph, pl):
    expr_parse = pl.struct(
        pl.col("syntax").struct.field("to_index").alias("arc"),
        pl.col("syntax").struct.field("rel").alias("rel"),
        pl.col("morph").struct.field("lemma").alias("word"),
    ).alias("parse")


    def get_entry(df, id_file: str, id_utterance: int, key: str | None = None):
        _df = df.filter(
            pl.col("id_file").str.contains(id_file),
            pl.col("id_utterance").eq(id_utterance),
        )
        if len(_df) == 0:
            raise KeyError(
                "No entry found for id_file={} and id_utterance={}".format(
                    id_file, id_utterance
                )
            )
        elif len(_df) > 1:
            raise KeyError(
                "Multiple entries found for id_file={} and id_utterance={}".format(
                    id_file, id_utterance
                )
            )
        return _df[0].item(0, key) if key is not None else _df[0]


    def draw(parse, verb):
        return draw_parse_graph(
            create_parse_graph(parse),
            highlight_node=lambda g, n, d: ("bold", "red")
            if d["label"] == verb
            else (None, None),
            highlight_edge=lambda g, u, v, d: (
                ("bold", "red")
                if g.nodes[u]["label"] == verb
                else ("bold", "blue")
                if g.nodes[v]["label"] == verb
                else (None, None)
            ),
        )

    return draw, expr_parse, get_entry


@app.cell
def _(mo, pl):
    map_dataset_annotation = {
        "2026-03-21": pl.read_parquet(
            "./var/pychildes/debug-annotation/pychildes-morphosyntax_annotation_20260321.20251024.parquet"
        ),
        "2026-03-19": pl.read_parquet(
            "./var/pychildes/debug-annotation/pychildes-morphosyntax_annotation_make-let_20260319.20251024.parquet"
        ),
        "2026-03-19-have": pl.read_parquet(
            "./var/pychildes/debug-annotation/pychildes-morphosyntax_annotation_have_20260319.20251024.parquet"
        ),
        "2026-02-25": pl.read_parquet(
            "./data/pychildes/pychildes-morphosyntax_annotation-20260225.20251024.parquet"
        ).filter(pl.col("features").list.len() > 0),
    }

    selector_dataset_annotation = mo.ui.dropdown(
        options=map_dataset_annotation,
        value="2026-03-21",
        label="Dataset version",
    )

    mo.vstack(
        [
            mo.md("## Statistics"),
            selector_dataset_annotation,
        ]
    )
    return map_dataset_annotation, selector_dataset_annotation


@app.cell
def _(selector_dataset_annotation):
    dataset_annotation = selector_dataset_annotation.value
    return (dataset_annotation,)


@app.cell
def _(dataset_annotation, pl):
    dataset_annotation_stats = dataset_annotation.filter(
        pl.col("features").list.contains("causative")
    ).select(
        "id_file",
        "id_utterance",
        "features",
        pl.col("featured_verbs").list.filter(
            pl.element().struct["feature"] == "causative"
        ),
    )
    return (dataset_annotation_stats,)


@app.cell
def _(dataset_annotation_stats, pl):
    (
        dataset_annotation_stats.select(
            pl.col("featured_verbs")
            .list.eval(
                pl.element().struct["extra"].str.json_path_match("$.support_verb")
            )
            .list.sort()
            .alias("support_verb")
        )
        .group_by(pl.col("support_verb"))
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    return


@app.cell
def _(VERB_TARGET_17, VERB_TARGET_5, dataset_annotation_stats, pl):
    (
        df_verb := (
            dataset_annotation_stats.select(
                pl.col("featured_verbs")
                .list.eval(pl.element().struct["verb"])
                .alias("verb")
                .list.explode()
            )
            .group_by("verb")
            .agg(pl.len().alias("count"))
            .with_columns(
                pl.col("verb")
                .replace_strict(
                    {
                        **{v: 2 for v in VERB_TARGET_17},
                        **{v: 1 for v in VERB_TARGET_5},
                    },
                    default=pl.lit(3),
                )
                .alias("interest_level")
            )
            .sort("interest_level", "count", descending=[False, True])
        )
    )
    return (df_verb,)


@app.cell
def _(VERB_TARGET_17, df_verb, mo):
    _list_missing_17 = list(set(VERB_TARGET_17) - set(df_verb["verb"]))

    mo.md(f"Missing Data: {', '.join(_list_missing_17)}")
    return


@app.cell
def _(dataset, map_dataset_annotation, mo, pl):
    df_addition = (
        map_dataset_annotation["2026-03-19"]
        .filter(pl.col("features").list.contains("causative"))
        .join(
            map_dataset_annotation["2026-02-25"].filter(
                pl.col("features").list.contains("causative")
            ),
            on=["id_file", "id_utterance"],
            how="anti",
        )
    )

    table_addition = mo.ui.table(
        dataset.join(
            df_addition,
            on=["id_file", "id_utterance"],
            how="inner",
        ).filter(pl.col("sentence").str.contains("make")),
        selection="single",
    )
    return (table_addition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diff
    """)
    return


@app.cell
def _(dataset, draw, expr_parse, get_entry, mo, pl, table_addition):
    if table_addition.value.height:
        _entry = get_entry(
            dataset,
            table_addition.value[0, "id_file"],
            table_addition.value[0, "id_utterance"],
        )

        img_addition_parse = draw(
            _entry.select(expr_parse)[0, "parse"],
            table_addition.value.select(
                pl.col("featured_verbs")
                .list.eval(pl.element().struct["verb"])
                .list.first()
            )[0, "featured_verbs"],
        )
    else:
        img_addition_parse = None

    mo.vstack(
        [
            mo.md("### Added Data on *make*"),
            table_addition,
        ]
    )
    return (img_addition_parse,)


@app.cell
def _(img_addition_parse):
    img_addition_parse
    return


@app.cell
def _(dataset, map_dataset_annotation, mo, pl):
    df_removal = (
        map_dataset_annotation["2026-02-25"]
        .filter(pl.col("features").list.contains("causative"))
        .join(
            map_dataset_annotation["2026-03-19"].filter(
                pl.col("features").list.contains("causative")
            ),
            on=["id_file", "id_utterance"],
            how="anti",
        )
    )

    table_removal = mo.ui.table(
        dataset.join(
            df_removal,
            on=["id_file", "id_utterance"],
            how="inner",
        ).filter(pl.col("sentence").str.contains("make")),
        selection="single",
    )
    return (table_removal,)


@app.cell
def _(dataset, draw, expr_parse, get_entry, mo, pl, table_removal):
    if table_removal.value.height:
        _entry = get_entry(
            dataset,
            table_removal.value[0, "id_file"],
            table_removal.value[0, "id_utterance"],
        )

        img_removal_parse = draw(
            _entry.select(expr_parse)[0, "parse"],
            table_removal.value.select(
                pl.col("featured_verbs")
                .list.eval(pl.element().struct["verb"])
                .list.first()
            )[0, "featured_verbs"],
        )
    else:
        img_removal_parse = None

    mo.vstack(
        [
            mo.md("### Removed Data on *make*"),
            table_removal,
        ]
    )
    return (img_removal_parse,)


@app.cell
def _(img_removal_parse):
    img_removal_parse
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Annotation
    """)
    return


@app.cell
def _(dataset, dataset_annotation, pl):
    dataset_annotation_causative = dataset.join(
        dataset_annotation.filter(pl.col("features").list.contains("causative")),
        on=["id_file", "id_utterance"],
    )
    return (dataset_annotation_causative,)


@app.cell
def _(mo):
    textbox_verb_filter = mo.ui.text(placeholder="", label="Verb filter")
    textbox_verb_filter
    return (textbox_verb_filter,)


@app.cell
def _(dataset_annotation_causative, mo, pl, textbox_verb_filter):
    _verb = textbox_verb_filter.value.strip()
    if _verb:
        _df_causative = dataset_annotation_causative.filter(
            pl.col("featured_verbs")
            .list.eval(pl.element().struct["verb"].eq(_verb))
            .list.any()
        )
    else:
        _df_causative = dataset_annotation_causative

    table_annotation = mo.ui.table(_df_causative, selection="single")

    mo.vstack(
        [
            mo.md(
                f"**{table_annotation.data.height} causative entry**"
                + f" of _{_verb}_"
                if _verb
                else ""
            ),
            table_annotation,
        ]
    )
    return (table_annotation,)


@app.cell
def _(dataset, draw, expr_parse, get_entry, pl, table_annotation):
    if table_annotation.value.height:
        _entry = get_entry(
            dataset,
            table_annotation.value[0, "id_file"],
            table_annotation.value[0, "id_utterance"],
        )

        img_annotation_parse = draw(
            _entry.select(expr_parse)[0, "parse"],
            table_annotation.value.select(
                pl.col("featured_verbs")
                .list.eval(pl.element().struct["verb"])
                .list.first()
            )[0, "featured_verbs"],
        )
    else:
        img_annotation_parse = None

    img_annotation_parse
    return


if __name__ == "__main__":
    app.run()
