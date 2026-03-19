# /// script
# [tool.marimo]
# runtime.autoreload = "lazy"
# display.custom_css = ["../../../../notebook.css"]
# ///

import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from functools import partial

    import marimo as mo
    import polars as pl

    return mo, os, partial, pl


@app.cell
def _():
    from verb_overgen.prep.lib.graph import create_parse_graph, draw_parse_graph
    from verb_overgen.prep.lib.lexical import LIST_CAUSATIVE_VERBS
    from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_verb
    from verb_overgen.prep.lib.syntactic import Parse
    from verb_overgen.train.lib.validate import VERB_TARGET_5

    return Parse, create_parse_graph, draw_parse_graph


@app.cell
def _(os):
    os.chdir(os.environ["PWD"])
    return


@app.cell
def _(pl):
    expr_parse = pl.struct(
        pl.col("syntax").struct.field("to_index").alias("arc"),
        pl.col("syntax").struct.field("rel").alias("rel"),
        pl.col("morph").struct.field("lemma").alias("word"),
    ).alias("parse")
    return (expr_parse,)


@app.cell
def _(expr_parse, pl):
    dataset_main = pl.read_parquet(
        "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"
    )
    dataset_annotation = pl.read_parquet(
        "./data/pychildes/pychildes-morphosyntax_annotation-20260221.20251024.parquet"
    )

    dataset = dataset_main.join(
        dataset_annotation, on=["id_file", "id_utterance"]
    ).with_columns(expr_parse)
    return (dataset,)


@app.cell
def _(create_parse_graph, draw_parse_graph, pl):
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

    return draw, get_entry


@app.cell
def dataframe_assert(pl):
    def assert_disjoint(
        df_1: pl.DataFrame, df_2: pl.DataFrame, *, on: str | list[str]
    ) -> None:
        """Assert no df_2 indices are in df_1."""
        overlap = df_1.join(df_2.select(on), on=on, how="semi")
        assert len(overlap) == 0, (
            f"Expected disjoint, but {len(overlap)} rows overlap on {on!r}"
        )


    def assert_subset(
        df_1: pl.DataFrame, df_2: pl.DataFrame, *, on: str | list[str]
    ) -> None:
        """Assert all df_2 indices are in df_1."""
        missing = df_2.join(df_1.select(on), on=on, how="anti")
        assert len(missing) == 0, (
            f"Expected df_2 ⊆ df_1, but {len(missing)} rows in df_2 are missing from df_1 on {on!r}"
        )

    return


@app.cell
def _(Parse):
    def comp_children(
        parse: Parse,
        *,
        target: str,
    ) -> list[str]:
        """Extract the comp children of the given support verb"""
        word, arc, rel = parse["word"], parse["arc"], parse["rel"]

        if target not in word:
            return []

        if "COMP" not in rel:
            return []

        # A map from support verb's lemma indices to their children's lemma indices
        target_children = {
            i1: [j for j, a1 in enumerate(arc) if a1 == i1]
            for i1, w in enumerate(word, 1)
            if w == target
        }

        # Array indices of the `COMP` children of the support verb
        indices_target_children = [
            [i for i in l if rel[i] == "COMP"] for l in target_children.values()
        ]

        # if verify_subj:
        #     # Array indices of nodes that have a `SUBJ` children.
        #     indices_has_subj = {to1 - 1 for r, to1 in zip(rel, arc) if r == "SUBJ"}

        #     # Filter the `COMP` children of the support verb to those that have a `SUBJ` child
        #     indices_target = [
        #         list(set(l).intersection(indices_has_subj)) for l in indices_target
        #     ]

        word_target_children = [
            word[i] for l in indices_target_children for i in l
        ]
        return word_target_children


    def has_subj(
        parse: Parse,
        *,
        target: str,
    ) -> bool:
        word, arc, rel = parse["word"], parse["arc"], parse["rel"]

        if target not in word:
            return False

        if "SUBJ" not in rel:
            return False

        target_children = {
            i1: [j for j, a1 in enumerate(arc) if a1 == i1]
            for i1, w in enumerate(word, 1)
            if w == target
        }

        return any(
            len([i for i in l if rel[i] == "SUBJ"])
            for l in target_children.values()
        )

    return (comp_children,)


@app.cell
def _(pl):
    def has_subj_somewhere_before(
        expr_parse: pl.Expr,
        *,
        target: str,
    ):
        return (
            expr_parse.struct.field("word")
            .list.eval(pl.element().eq(target).arg_true())
            .list.last()
            > expr_parse.struct.field("rel")
            .list.eval(pl.element().eq("SUBJ").arg_true())
            .list.first()
        ).fill_null(False)

    return


@app.cell
def func_has_subj_in_parent(Parse):
    def has_subj_in_parent(
        parse: Parse,
        *,
        target: str,
    ) -> bool:
        """Recursively find `target` and `COMP`-ancestors to see if there's an ancestor with a `SUBJ` children."""
        list_word, list_arc, list_rel = parse["word"], parse["arc"], parse["rel"]

        if target not in list_word:
            return False

        if "SUBJ" not in list_rel:
            return False

        list_index1_target = [
            i1 for i1, w in enumerate(list_word, 1) if w == target
        ]

        for index1_target in list_index1_target:
            # We first check if target has any subject
            index1: int = index1_target

            # If index1 exists
            while index1:
                # If a node is index1's `SUBJ` children
                index1_has_subject = any(
                    to1 == index1 and rel == "SUBJ"
                    for to1, rel in zip(list_arc, list_rel)
                )
                if index1_has_subject:
                    # Then we decide this rule satisifies
                    return True

                # Otherwise check its `COMP`-parent, or set to 0
                index1 = (
                    list_arc[index1 - 1] if list_rel[index1 - 1] == "COMP" else 0
                )

        return False

    return (has_subj_in_parent,)


@app.cell
def _(comp_children, expr_parse, has_subj_in_parent, partial, pl):
    PERIPHERAL_VERB = "let"

    MAP_RULE = {
        "rule_has_let": pl.col("morph")
        .struct.field("lemma")
        .list.contains(item=PERIPHERAL_VERB),
        "rule_not_imp_let_us": ~pl.col("content_raw").str.contains(
            r"\b[Ll]et's\b"
        ),
        "rule_not_let_go": ~pl.col("content_raw").str.contains(r"\b[Ll]et go\b"),
        "rule_has_comp_children": expr_parse.map_elements(
            partial(comp_children, target=PERIPHERAL_VERB),
            pl.List(pl.String),
        ).list.len()
        > 0,
        "rule_has_let_subj": expr_parse.map_elements(
            partial(has_subj_in_parent, target=PERIPHERAL_VERB),
            pl.Boolean,
        ),
        # "rule_has_subj": expr_parse.pipe(has_subj_somewhere_before, target=PERIPHERAL_VERB)
    }
    return MAP_RULE, PERIPHERAL_VERB


@app.cell
def _():
    SET_RULE_CONFIDENT = {
        "rule_has_let",
    }

    SET_RULE_SPURIOUS = set([
        "rule_has_let_subj",
    ])
    return SET_RULE_CONFIDENT, SET_RULE_SPURIOUS


@app.cell
def _(MAP_RULE, SET_RULE_CONFIDENT, dataset):
    dataset_prefiltered = dataset.filter(
        [
            _rule
            for _rule_name, _rule in MAP_RULE.items()
            if _rule_name in SET_RULE_CONFIDENT
        ]
    )
    return (dataset_prefiltered,)


@app.cell
def _(MAP_RULE, dataset_prefiltered):
    dataset_annotated = dataset_prefiltered.with_columns(
        **MAP_RULE,
    )
    return (dataset_annotated,)


@app.cell
def _(MAP_RULE, SET_RULE_CONFIDENT, mo, pl):
    EXPR_RULE_INCONFIDENT = pl.col(MAP_RULE.keys() - SET_RULE_CONFIDENT)

    EXPR_CORE_COLUMN = [
        pl.selectors.starts_with("id_"),
        pl.col("sentence"),
        pl.col("content_raw"),
        EXPR_RULE_INCONFIDENT,
    ]

    checkbox_should_sample = mo.ui.checkbox(label="Should Sample")
    return EXPR_CORE_COLUMN, checkbox_should_sample


@app.cell
def _(
    EXPR_CORE_COLUMN,
    MAP_RULE,
    SET_RULE_CONFIDENT,
    SET_RULE_SPURIOUS,
    checkbox_should_sample,
    dataset_annotated,
    mo,
    pl,
):
    dataset_annotated_filtered = dataset_annotated.filter(
        pl.all_horizontal(
            pl.col(MAP_RULE.keys() - SET_RULE_CONFIDENT - SET_RULE_SPURIOUS)
        ),
        ~pl.col("rule_has_let_subj"),
        pl.col("sentence").str.contains("go to let"),
    ).select(*EXPR_CORE_COLUMN)

    _df = (
        dataset_annotated_filtered.sample(50, seed=42)
        if checkbox_should_sample.value and dataset_annotated_filtered.height > 50
        else dataset_annotated_filtered
    )

    table_annotated = mo.ui.table(_df, selection="multi")

    mo.vstack(
        [
            checkbox_should_sample,
            dataset_annotated_filtered.height,
            table_annotated,
        ]
    )
    return dataset_annotated_filtered, table_annotated


@app.cell
def _(dataset_annotated_filtered, pl, table_annotated):
    _df = table_annotated.value
    _df_n = dataset_annotated_filtered.join(table_annotated.value, on=["id_file", "id_utterance"], how="anti")

    df_let_addendum = _df.select(pl.selectors.starts_with("id"), pl.lit(True).alias("rule_is"))

    df_let_addendum
    return (df_let_addendum,)


@app.cell
def _(df_let_addendum):
    df_let_addendum.write_csv("./var/pychildes/rules/causative_let_addendum.csv")
    return


@app.cell
def _(dataset_annotated_filtered, pl):
    # This is a playground code

    _ser = dataset_annotated_filtered.filter(
        pl.col("id_file") == "Eng-UK/Belfast/David/020122",
    ).select(
        pl.struct("id_file", "id_utterance")
    ).to_series()

    dataset_annotated_filtered.select(
        pl.struct("id_file", "id_utterance").replace_strict(_ser, pl.lit(True), default=None).alias("rule_is"),
        pl.row_index().mod(2).eq(1).alias("rule_odd")
    ).with_columns(
        pl.coalesce("rule_is", "rule_odd").alias("rule_or")
    )
    return


@app.cell
def _(PERIPHERAL_VERB, dataset, draw, get_entry, table_annotated):
    _out = [
        draw(
            get_entry(
                dataset,
                table_annotated.value[_idx, "id_file"],
                table_annotated.value[_idx, "id_utterance"],
                "parse",
            ),
            PERIPHERAL_VERB,
        )
        for _idx in range(table_annotated.value.height)
    ]

    _out if _out else None
    return


@app.cell
def _(dataset, draw, get_entry):
    draw(
        get_entry(
            dataset,
            "Eng-NA/Brent/f1/001106",
            925,
            "parse",
        ),
        "ring",
    )
    return


@app.cell
def locate_false_negative(
    EXPR_CORE_COLUMN,
    PERIPHERAL_VERB,
    dataset,
    dataset_annotated_filtered,
    pl,
):
    dataset_negative = (
        dataset.filter(
            pl.col("morph")
            .struct.field("lemma")
            .list.contains(item=PERIPHERAL_VERB),
        )
        .join(
            dataset_annotated_filtered.select(pl.selectors.starts_with("id_")),
            on=["id_file", "id_utterance"],
            how="anti",
        )
        .select(*EXPR_CORE_COLUMN)
    )

    dataset_negative.sample(50, seed=42)
    return (dataset_negative,)


@app.cell
def _(dataset_negative, pl):
    dataset_negative.filter(pl.col("sentence").str.contains("be go to let"))
    return


if __name__ == "__main__":
    app.run()
