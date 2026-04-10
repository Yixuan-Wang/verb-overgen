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
    import altair as alt


    os.chdir(os.environ["PWD"])
    return alt, pl


@app.cell
def _():
    from verb_overgen.common.features import (
        VERB_TARGET_5,
        VERB_TARGET_17,
        VERB_TARGET_EXTENSION,
    )

    return VERB_TARGET_17, VERB_TARGET_5, VERB_TARGET_EXTENSION


@app.cell
def _(pl):
    dataset = pl.read_parquet(
        "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"
    )
    return (dataset,)


@app.cell
def _(pl):
    dataset_annotation = pl.read_parquet(
        "./var/pychildes/debug-annotation/pychildes-morphosyntax_annotation_20260321.20251024.parquet"
    )
    return (dataset_annotation,)


@app.cell
def _(dataset, pl):
    dataset_lemma_pos = (
        dataset.lazy()
        .select(
            "id_file",
            "id_utterance",
            pl.col("morph").struct.field("pos"),
            pl.col("morph").struct.field("lemma"),
        )
        .explode(["pos", "lemma"])
        .group_by("pos")
        .agg(
            pl.col("lemma").value_counts(),
            pl.len().alias("count"),
        )
        .with_columns(pl.col("lemma").list.len().alias("n_unique_lemma"))
        .sort(["count", "n_unique_lemma"], descending=[True, False])
        .collect()
    )
    dataset_lemma_pos
    return (dataset_lemma_pos,)


@app.cell
def _(dataset_lemma_pos, pl):
    df_verblike_lemma = (
        dataset_lemma_pos.select("pos", "lemma")
        .explode("lemma")
        .unnest("lemma")
        .group_by(
            "lemma",
            pl.col("pos").is_in(["v", "part"]).alias("is_verb"),
        )
        .agg(
            pl.col("pos"),
            pl.col("count"),
        )
        .group_by("lemma")
        .agg(
            pl.col("is_verb"),
            pl.col("is_verb").any().alias("has_verb"),
            pl.col("pos"),
            pl.col("count"),
        )
        .filter(pl.col("has_verb"))
        .select(pl.all().exclude("has_verb"))
        .explode("is_verb", "pos", "count")
        .with_columns(pl.col("count").list.sum().alias("total_count"))
        .pivot(
            index="lemma",
            on="is_verb",
            values=["total_count"],
        )
        .select(
            "lemma",
            pl.col("true").alias("verb_like_count"),
            pl.col("false").alias("non_verb_like_count"),
        )
        .sort("verb_like_count", descending=True)
    )
    df_verblike_lemma
    return (df_verblike_lemma,)


@app.cell
def _(dataset_annotation, pl):
    dataset_annotation_causative = dataset_annotation.filter(
        pl.col("features").list.contains("causative")
    ).select(
        "id_file",
        "id_utterance",
        "features",
        pl.col("featured_verbs").list.filter(
            pl.element().struct["feature"] == "causative"
        ),
    )
    dataset_annotation_causative
    return (dataset_annotation_causative,)


@app.cell
def _(dataset_annotation_causative, pl):
    df_causative_verb_count = (
        dataset_annotation_causative.select(
            pl.col("featured_verbs")
            .list.eval(pl.element().struct["verb"])
            .alias("verb")
            .list.explode()
        )
        .group_by("verb")
        .agg(pl.len().alias("causative_count"))
        .sort("causative_count", descending=True)
    )
    df_causative_verb_count
    return (df_causative_verb_count,)


@app.cell
def _(
    VERB_TARGET_17,
    VERB_TARGET_5,
    VERB_TARGET_EXTENSION,
    df_causative_verb_count,
    df_verblike_lemma,
    pl,
):
    df_combined = (
        df_verblike_lemma.join(
            df_causative_verb_count,
            left_on="lemma",
            right_on="verb",
            how="left",
        )
        .with_columns(pl.col("causative_count").fill_null(0))
        .rename(
            {
                "lemma": "verb",
                "verb_like_count": "count_verb_like",
                "non_verb_like_count": "count_non_verb_like",
                "causative_count": "count_causative",
            }
        )
        .with_columns(
            pl.col("verb")
            .replace_strict(
                {
                    **{
                        v: 3
                        for v in VERB_TARGET_EXTENSION[
                            "extended_intransitive_only"
                        ]
                    },
                    # These cannot be reordered.
                    **{v: 2 for v in VERB_TARGET_17},
                    **{v: 1 for v in VERB_TARGET_5},
                },
                default=pl.lit(4),
            )
            .alias("interest_level")
        )
        .sort("interest_level", "count_verb_like", descending=[False, True])
    )
    df_combined
    return (df_combined,)


@app.cell
def _(df_combined, pl):
    df_finder = df_combined.select(
        pl.col("verb", "interest_level"),
        pl.col("count_causative").alias("abs_preemption"),
        pl.col("count_verb_like").alias("abs_entrenchment"),
        pl.col("count_causative")
        .truediv(pl.col("count_verb_like"))
        .alias("rel_preemption"),
        pl.col("count_non_verb_like")
        .truediv(pl.col("count_verb_like"))
        .alias("rel_confounding"),
    )

    df_finder
    return (df_finder,)


@app.cell
def _():
    VERBS_FINAL = [
                "laugh", "sing", "sleep", "smile", "cry",
                "go", "sneeze", "blink", "look", "rise", "glow",
            ]
    return (VERBS_FINAL,)


@app.cell
def _(VERBS_FINAL, alt, df_finder, pl):
    _chart = (
        alt.Chart(df_finder.filter(
            pl.col("interest_level") < 4,
            pl.col("verb").is_in(VERBS_FINAL)
        ))
        .mark_circle()
        .encode(
            x=alt.X(
                "abs_entrenchment:Q",
                title="Entrenchment (verb-like count)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "abs_preemption:Q",
                title="Preemption (causative count)",
                scale=alt.Scale(type="symlog", constant=1),
                axis=alt.Axis(grid=False),
            ),
            color=alt.Color("interest_level:N", title="Interest level"),
            tooltip=[
                "verb:N",
                "abs_preemption:Q",
                "abs_entrenchment:Q",
                "rel_preemption:Q",
                "rel_confounding:Q",
                "interest_level:N",
            ],
        )
    )
    _chart
    return


@app.cell
def _(VERBS_FINAL, alt, df_finder, pl):
    _chart = (
        alt.Chart(df_finder.filter(
            pl.col("interest_level") < 4,
            # pl.col("verb").is_in(VERBS_FINAL)
        ).with_columns(
            pl.col("verb").is_in(VERBS_FINAL).alias("selected"),
        ))
        .mark_circle()
        .encode(
            x=alt.X(
                "abs_preemption:Q",
                title="Preemption (absolute)",
                scale=alt.Scale(type="symlog", constant=1),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "rel_preemption",
                title="Preemption (relative)",
                axis=alt.Axis(grid=False),
            ),
            color=alt.Color("selected:N", title="Selected"),
            tooltip=[
                "verb:N",
                "abs_preemption",
                "abs_entrenchment",
                "rel_preemption",
                "rel_confounding",
                "interest_level:N",
            ],
        )
    )
    _chart
    return


@app.cell
def _():
    # df_combined.write_parquet("./var/results/verb_stats.parquet")
    return


@app.cell
def _(df_finder, pl):
    print(
        df_finder.sort("abs_preemption", descending=True)
        .filter(pl.col("interest_level") == 3)
        .head(100)
        .select("verb")
        .to_series()
        .to_list()
    )
    return


if __name__ == "__main__":
    app.run()
