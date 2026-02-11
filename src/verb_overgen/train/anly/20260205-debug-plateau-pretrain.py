# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import gc

    import altair as alt
    import marimo as mo
    import polars as pl
    import sh
    import torch
    import transformers as hf

    return alt, gc, hf, mo, pl, torch


@app.cell
def _():
    import numpy as np

    return


@app.cell
def _():
    from verb_overgen.train.lib.validate import perplexity

    return (perplexity,)


@app.cell
def _():
    import os

    os.chdir(os.environ["PWD"])
    return


@app.cell
def _():
    RUN_MAPPING = {
        "copper-firebrand-62": {
            "kind": "baseline",
        },
        "comfy-serenity-64": {
            "kind": "tune-pretrain",
        },
        "dazzling-deluge-63": {"kind": "freeze-pretrain"},
    }
    return (RUN_MAPPING,)


@app.cell
def _(pl):
    dataset_verb_stat = pl.concat([
            pl.read_csv("./data/testset/gen/causative_verb_counts_supplementary.csv"),
            pl.read_csv("./data/testset/gen/causative_verb_counts.csv")
        ]).rename({"lemma": "verb"})
    return (dataset_verb_stat,)


@app.cell
def _(RUN_MAPPING, gc, hf, pl, torch):
    def gather_stats():
        pretrained_model = hf.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

        # Collect statistics for all models
        stats_records = []

        # Add pretrained model statistics
        for param_name, param in pretrained_model.named_parameters():
            clean_name = param_name.replace("transformer.", "")
            stats_records.append(
                {
                    "kind": "pretrain",
                    "param_name": clean_name,
                    "shape": list(param.shape),
                    "abs_mean": param.abs().mean().item(),
                    "abs_std": param.abs().std().item(),
                }
            )

        # Load and analyze checkpoints
        for run, info in RUN_MAPPING.items():
            path_ckpt = f"./var/train/20260128/{run}/checkpoints/last.ckpt"

            # Load checkpoint
            checkpoint = torch.load(path_ckpt, map_location="cpu", weights_only=False)
            state_dict = checkpoint["state_dict"]

            for param_name, param in state_dict.items():
                clean_name = param_name.replace("model.", "").replace("transformer.", "")

                stats_records.append(
                    {
                        "kind": info["kind"],
                        "param_name": clean_name,
                        "shape": list(param.shape),
                        "abs_mean": param.abs().mean().item(),
                        "abs_std": param.abs().std().item(),
                    }
                )

            del checkpoint
            gc.collect()

        # %%
        # Create Polars dataframe
        df_stats = pl.DataFrame(stats_records, schema_overrides={
            "shape": pl.List(pl.Int64),
        })
        return df_stats

    return (gather_stats,)


@app.cell
def _(gather_stats):
    df_stats = gather_stats()
    df_stats
    return (df_stats,)


@app.cell
def _(df_stats):
    df_stats_mean = df_stats.pivot(on="kind", index="param_name", values=["abs_mean"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Landscape of Parameters
    """)
    return


@app.cell
def _(mo):
    dropdown_param_stat_type = mo.ui.dropdown(["abs_mean", "abs_std"], label="Parameter Statistic Type", value="abs_mean")
    return (dropdown_param_stat_type,)


@app.cell
def _(alt, df_stats, dropdown_param_stat_type, mo, pl):
    _df = df_stats.with_columns(
        pl.col("param_name").str.replace("(h\.\d+)\.(.*)", r"$2.$1").alias("param_name")
    )

    _chart = alt.Chart(_df).mark_bar().encode(
        x=alt.X(f"{dropdown_param_stat_type.value}:Q", title="Absolute Mean" if dropdown_param_stat_type.value == "abs_mean" else "Absolute Std Dev"),
        y=alt.Y("param_name:N", title="Parameter Name"),
        color=alt.Color("kind:N", title="Model Type"),
        yOffset=alt.YOffset("kind:N"),
        tooltip=["kind", "param_name", "abs_mean", "abs_std"],
    ).properties(
        title=f"Absolute {'Mean' if dropdown_param_stat_type.value == 'abs_mean' else 'Std Dev'} of Parameters by Model Type"
    )

    mo.vstack([
        dropdown_param_stat_type,
        _chart,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Testset Results
    """)
    return


@app.cell
def _(RUN_MAPPING, perplexity, pl):
    def gather_testset_results():
        records = []
        for run_name in RUN_MAPPING.keys():
            df_run = pl.read_parquet(f"./var/train/20260128/{run_name}/test/").with_columns(
                pl.lit(run_name).alias("run_name"),
                pl.lit(RUN_MAPPING[run_name]["kind"]).alias("kind")
            )
            records.append(df_run)

        dataset_run = pl.concat(records)
        dataset_run = dataset_run.with_columns(
            pl.col("token_prob").pipe(perplexity).alias("perplexity")
        )
        return dataset_run

    return (gather_testset_results,)


@app.cell
def _(gather_testset_results, pl):
    dataset_run = gather_testset_results().filter(pl.col("step") != 0)
    dataset_input = pl.read_parquet("./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet")

    assert (
        dataset_run.select(pl.col("index_sentence").unique().sort()).to_series()
        == dataset_input.select(pl.col("index_sentence").unique().sort()).to_series()
    ).all()

    dataset_input = dataset_input.with_columns(
        pl.col("input_ids").list.slice(0, pl.col("input_ids").list.len() - 1).alias("input_ids")
    )


    dataset_run = dataset_run.join(dataset_input.filter(pl.col(name="kind") != "sentence_context").select("index_sentence"), on="index_sentence", how="semi")

    dataset = dataset_run.join(
            dataset_input.select(
                pl.col("index_sentence"),
                pl.col("index").alias("index_pair"),
                pl.col("verb", "vet"),
                pl.col("kind").alias("sentence_kind"),
                pl.col("sentence"),
            ),
            on="index_sentence",
            how="left",
            coalesce=True,
        )

    dataset
    return dataset, dataset_input, dataset_run


@app.cell
def _(alt, dataset, dataset_verb_stat, pl):
    _df = dataset.filter(pl.col("epoch") == 2).with_columns(
        pl.col("perplexity").log().alias("nll"),
    )

    _df = _df.group_by("kind", "verb").agg(
        pl.mean("nll").alias("mean_nll"),
        pl.std("nll").alias("std_nll"),
    )

    alt.Chart(_df).mark_bar().encode(
        x=alt.X("mean_nll:Q", title="Mean NLL"),
        y=alt.Y("verb:N", title="Verb", sort=dataset_verb_stat.sort("count_is_verb", descending=True)["verb"].to_list()),
        color=alt.Color("kind:N", title="Model Type"),
        yOffset=alt.YOffset("kind:N"),
    )
    return


@app.cell
def _(pl):
    dataset_token_count = pl.read_parquet("./var/pychildes/ngram/token_count.parquet")
    return (dataset_token_count,)


@app.cell
def _(dataset_token_count, pl):
    dataset_token_freq = dataset_token_count.select(
        pl.col("input_ids").alias("token"),
        pl.col("count").truediv(pl.col("count").sum()).alias("token_freq")
    )

    map_token_freq = dict(zip(
        dataset_token_freq["token"],
        dataset_token_freq["token_freq"],
    ))
    return (map_token_freq,)


@app.cell
def _(dataset_input, map_token_freq, pl):
    dataset_input_token_freq = dataset_input.select(
        pl.col("index_sentence"),
        pl.col("input_ids")
        .list.eval(pl.element().map_elements(map_token_freq.get, pl.Float64)).alias("token_freq")
    )
    return (dataset_input_token_freq,)


@app.cell
def _(dataset, dataset_input_token_freq, dataset_run, pl):
    def slor(token_prob: pl.Expr, token_freq: pl.Expr):
        return (
            (token_prob.list.eval(pl.element().cast(pl.Float64).log()).list.sum())
            .sub(token_freq.list.eval(pl.element().cast(pl.Float64).log()).list.sum())
            .truediv(token_prob.list.len())
            .alias("slor")
        )


    dataset_slor = dataset_run.join(dataset_input_token_freq.select("index_sentence", "token_freq"), on="index_sentence", how="left").select(
        slor(pl.col("token_prob"), pl.col("token_freq"))
    )

    dataset_with_slor = pl.concat([dataset, dataset_slor], how="horizontal")
    return (dataset_with_slor,)


@app.cell
def _(mo):
    mo.md(r"""
    $\mathrm{SLOR}(x) = \frac{1}{|x|} \left( \log P(x) - \log P_\mathrm{unigram}(x) \right)$
    """)
    return


@app.cell
def _(alt, dataset_verb_stat, dataset_with_slor, pl):
    _df = dataset_with_slor.filter(pl.col("epoch") == 2)

    _df = _df.group_by("kind", "verb").agg(
        pl.mean("slor").alias("mean_slor"),
        pl.std("slor").alias("std_slor"),
    )

    alt.Chart(_df).mark_bar().encode(
        x=alt.X("mean_slor:Q", title="Mean SLOR"),
        y=alt.Y("verb:N", title="Verb", sort=dataset_verb_stat.sort("count_is_verb", descending=True)["verb"].to_list()),
        color=alt.Color("kind:N", title="Model Type"),
        yOffset=alt.YOffset("kind:N"),
    )
    return


@app.cell
def _(dataset_with_slor, pl):
    dataset_transitive = dataset_with_slor.filter(pl.col("sentence_kind") == "sentence_transitive")
    return


if __name__ == "__main__":
    app.run()
