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
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    ```markdown
    - Finished eval set refinement and test set generation.
    - Migrated infra to Vector's cluster.
    - Plan: Training receipes
      1. Full training data
      2. Remove preemption
      3. Remove entrenchment - preemption

      1. All candidate verbs removal
      2. Remove individual candidate verbs (e.g. laugh)
    ```
    """)
    return


@app.cell
def _(pl):
    dataset_testset = pl.read_parquet("/home/tomyxw/Projects/2025/verb-overgen/verb-overgen/data/testset/testset_llm_gen_supplementary-postprocessed.20260108T2325.parquet")
    dataset_testset = dataset_testset.select(
        pl.selectors.exclude(pl.selectors.ends_with("right"))
    )
    return (dataset_testset,)


@app.cell
def _(dataset_testset, mo, pl):
    list_verb = dataset_testset.select(pl.col("verb")).unique().sort("verb").to_series().to_list()
    sel_verb = mo.ui.dropdown(list_verb, list_verb[0])

    checkbox_show_context = mo.ui.checkbox(label="Show context sentences", value=False)
    return checkbox_show_context, sel_verb


@app.cell
def _(mo):
    mo.md(r"""
    How the test set is generated?

    - Provide a subject and a verb, then 25 object candidates (nouns mixed with human names).
    - All words are sampled from training set. Guarantee to be within vocab.
      (External sources have too few cases of the target verbs, and are likely out of domain)
    - The model is asked to pick an object and then generate a "context" sentence and then a "main" sentence containing the `make` construction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Some different verbs within the Bidgood set.

    ## laugh

    Some verbs might not apply to non-human objects?

    1697: Tony makes the duck laugh. <-> Tony laughs the duck.

    ## fall, rise, die

    Some verbs have lexical causative alternations that doesn't fit into the "make" construction.

    900: Helen makes the bear fall. <-> Helen falls the bear.
        (Helen makes the bear drop. <-> Helen drops the bear.)

    rise <-> raise
    die  <-> kill


    ## eat, wait

    The preempted candidate construction might mean something else.

    2700: Luke makes Henry wait. <-> Luke waits Henry.
          (Henry waits. <-> Luke waits.)

    677:  Rosie makes the pig eat. <-> Rosie eats the pig.
          (The pig eats. <-> Rosie eats the pig.)
          (eat <-> feed)

    ## go

    How to handle residuals in preempted candidate constructions?

    1337: Thomas makes the kitty go home. <-> Thomas goes the kitty home.
          (or, Thomas goes home the kitty?)

    ## leak

    No good candidate objects. Seems the LLM confused it with "spill" or "break".
    """)
    return


@app.cell
def _(checkbox_show_context, dataset_testset, mo, pl, sel_verb):
    _df = dataset_testset.filter(pl.col("verb") == sel_verb.value)

    _df = _df.select(
        pl.col("index"),
        pl.col("verb"),
        pl.selectors.starts_with("sentence_")
    )

    if not checkbox_show_context.value:
        _df = _df.select(
            pl.all().exclude("sentence_context")
        )

    mo.vstack([
        sel_verb,
        checkbox_show_context,
        _df,
    ])
    return


@app.cell
def _(dataset_testset):
    dataset_testset
    return


@app.cell
def _(dataset_testset, pl):
    expr_lemma = lambda expr: expr.struct["lemma"].list.join(" ")

    dataset_testset.select(
        pl.col("index"),
        pl.col("verb"),
        pl.col("sentence_context"),
        pl.col("sentence_target"),
        pl.col("sentence_target_alt"),
        pl.col("llm_gen").struct["output_reasoning"].alias("llm_gen_reasoning"),
        pl.col("roberta_context").pipe(expr_lemma).alias("lemmatized_sentence_target"),
        pl.col("roberta_target").pipe(expr_lemma).alias("lemmatized_sentence_context"),
        pl.col("roberta_target_alt").pipe(expr_lemma).alias("lemmatized_sentence_target_alt"),
    ).write_csv("dataset_testset_supplementary.csv")
    return


if __name__ == "__main__":
    app.run()
