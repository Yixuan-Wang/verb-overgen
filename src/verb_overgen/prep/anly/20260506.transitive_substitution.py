# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import os
    from typing import cast

    import marimo as mo
    import polars as pl

    os.chdir(os.environ["PWD"])


@app.cell
def _():
    from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED

    return (FEATURE_CAUSATIVE,)


@app.cell
def _():
    dataset_main = pl.scan_parquet(
        "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"
    )
    dataset_annotation = pl.scan_parquet(
         "./data/pychildes/pychildes-morphosyntax_annotation-20260402.20251024.parquet"
    )
    return dataset_annotation, dataset_main


@app.cell
def _(dataset_annotation):
    dataset_annotation.schema
    return


@app.cell
def _():
    VERB_OF_INTEREST = "sing"
    return (VERB_OF_INTEREST,)


@app.cell
def _(FEATURE_CAUSATIVE, VERB_OF_INTEREST, dataset_annotation, dataset_main):
    dataset = cast(
        pl.DataFrame,
        dataset_main.join(
            dataset_annotation,
            on=["id_file", "id_utterance"],
        )
        .filter(
            pl.col("features").list.contains(FEATURE_CAUSATIVE),
            pl.col("featured_verbs")
            .list.eval((pl.element().struct["verb"] == VERB_OF_INTEREST) & (pl.element().struct["feature"] == FEATURE_CAUSATIVE))
            .list.any(),
        )
        .collect()
    )
    return (dataset,)


@app.cell
def _(VERB_OF_INTEREST, dataset):
    dataset.filter(
        pl.col("sentence").str.contains(
            rf"(make|get|let|have|cause|force) (.+?) to {VERB_OF_INTEREST}"
        )
    )
    return


@app.function
def substitute_causative(
    sentence: str,
    target: str,
    peripheral_verbs: frozenset[str] = frozenset(
        {"make", "get", "let", "have", "cause", "force"}
    ),
) -> str:
    tokens = sentence.split()
    out: list[str] = []
    for tok in tokens:
        if tok == target:
            obj_end = len(out)
            # Drop a periphrastic "to" preceding the target — the lexical
            # causative form takes a bare object, not an infinitival complement.
            if obj_end > 0 and out[obj_end - 1] == "to":
                obj_end -= 1
            peripheral_idx = next(
                (j for j in range(obj_end - 1, -1, -1) if out[j] in peripheral_verbs),
                None,
            )
            if peripheral_idx is not None:
                out = out[:peripheral_idx] + [target] + out[peripheral_idx + 1 : obj_end]
                continue
        out.append(tok)
    return " ".join(out)


@app.cell
def _(VERB_OF_INTEREST, dataset):
    _regex = rf"\b(make|get|let|have|cause|force) (.+?) ((?:to )?{VERB_OF_INTEREST})"
    _causative_pattern = r"\b(?:make|get|let|have|cause|force)\b"
    _comparison = dataset.select(
        pl.col("sentence"),
        pl.col("sentence").str.replace_all(_regex, r"$3 $2").alias("sentence_regex"),
        pl.col("sentence")
        .map_elements(
            lambda s: substitute_causative(s, VERB_OF_INTEREST),
            return_dtype=pl.String,
        )
        .alias("sentence_func"),
        pl.col("sentence")
        .str.count_matches(_causative_pattern)
        .alias("n_matrix_verb"),
    )

    mo.output.append(_comparison.filter(pl.col("sentence_regex") != pl.col("sentence_func")))
    return


@app.cell
def _(VERB_OF_INTEREST, dataset):
    dataset.select(
        pl.col("sentence"),
        pl.col("sentence")
        .map_elements(
            lambda s: substitute_causative(s, VERB_OF_INTEREST),
            return_dtype=pl.String,
        )
        .alias("sentence_sub"),
    )
    return


if __name__ == "__main__":
    app.run()
