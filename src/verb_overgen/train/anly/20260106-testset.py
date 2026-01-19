# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import seaborn as sns
    import altair as alt
    import numpy as np
    import scipy.stats as stats

    alt.data_transformers.enable("vegafusion")
    return alt, mo, np, pl, stats


@app.cell
def _(pl):
    dataset_test_result = pl.read_parquet("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/var/train/20251230/celestial-mountain-7/test")
    dataset_input = pl.read_parquet("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet")
    return dataset_input, dataset_test_result


@app.cell
def _(dataset_input, dataset_test_result, mo):
    mo.vstack([
        dataset_input.head(5),
        dataset_test_result.head(5),
    ])
    return


@app.cell
def _(dataset_input, dataset_test_result, np, pl):
    def ppl(token_probs):
        nlls = -np.log(token_probs)
        mean_nll = np.mean(nlls)
        ppl = np.exp(mean_nll)
        return ppl

    def batch_ppl(token_probs_list):
        return pl.Series("ppl", [ppl(token_probs.to_numpy()) for token_probs in token_probs_list])

    dataset = dataset_test_result.join(dataset_input, on="index_sentence", how="left")
    dataset = dataset.filter(pl.col("kind") != "sentence_context").with_columns(
        pl.col("token_prob").map_batches(batch_ppl, pl.Float32).alias("ppl")
    )
    return (dataset,)


@app.cell
def _(pl):
    dataset_verb_stat = pl.concat([
        pl.read_csv("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/data/testset/gen/causative_verb_counts_supplementary.csv"),
        pl.read_csv("/home/tomyxw/projects/2025/verb-overgen/verb-overgen/data/testset/gen/causative_verb_counts.csv")
    ]).rename({"lemma": "verb"})
    return (dataset_verb_stat,)


@app.cell
def _(mo):
    mo.md(r"""
    $$
    \mathrm{ppl} \triangleq \exp\left(-\frac{1}{n} \sum_{i=1}^n \log p(x_i | x_{<i})\right)
    $$

    `<BOS>` and `<EOS>` are added to the input and `<EOS>` is considered when calculating the perplexity.
    """)
    return


@app.cell
def _(dataset, pl):
    dataset_ppl = dataset.group_by("step", "index").agg(
        pl.col("verb").unique().item(),
        pl.col("vet").unique().item(),
        pl.col("ppl").filter(pl.col("kind") == "sentence_causative").item().alias("ppl_causative"),
        pl.col("ppl").filter(pl.col("kind") == "sentence_transitive").item().alias("ppl_transitive"),
    ).sort(["step", "verb", "index"])

    dataset_ppl
    return (dataset_ppl,)


@app.cell
def _(dataset_ppl, mo):
    list_step = dataset_ppl.select("step").unique().to_series().to_list()
    dropdown_step = mo.ui.dropdown({ str(k): v for k, v in enumerate(list_step)}, value="0", label="Epoch") # pyright: ignore

    list_verb_order = ["alphabetical", "count", "count_verb", "count_causative"]
    dropdown_verb_order = mo.ui.dropdown(list_verb_order, value="alphabetical", label="Verb Order") # pyright: ignore
    return dropdown_step, dropdown_verb_order, list_step


@app.cell
def _(
    alt,
    dataset_ppl,
    dataset_verb_stat,
    dropdown_step,
    dropdown_verb_order,
    mo,
    pl,
):
    if dropdown_verb_order.value == "alphabetical":
        dataset_verb_order = dataset_verb_stat.sort("verb")
    elif dropdown_verb_order.value == "count":
        dataset_verb_order = dataset_verb_stat.sort("count", descending=True)
    elif dropdown_verb_order.value == "count_verb":
        dataset_verb_order = dataset_verb_stat.fill_null(0).sort("count_is_verb", descending=True)
    elif dropdown_verb_order.value == "count_causative":
        dataset_verb_order = dataset_verb_stat.fill_null(0).sort(pl.col("count_causative") / pl.col("count_is_verb"), descending=True)

    _df = (
        dataset_ppl.filter(
            pl.col("step") == dropdown_step.value
        ).unpivot(
            on=["ppl_causative", "ppl_transitive"],
            index=["step", "index", "verb", "vet"],
            variable_name="kind",
            value_name="ppl"
        ).with_columns(
            pl.col("kind").str.replace("ppl_", "")
        ).join(
            dataset_verb_order,
            on="verb",
            how="inner"
        )
    )

    chart = (
        alt.Chart(_df)
        .mark_boxplot()
        .encode(
            x=alt.X(
                "verb:N",
                title="Verb",
                axis=alt.Axis(labelAngle=-45),
                scale=alt.Scale(domain=dataset_verb_order["verb"].to_list()),
            ),
            y=alt.Y("ppl:Q", title="Perplexity"),
            color=alt.Color("kind:N", title="Kind"),
            xOffset=alt.XOffset("kind:N"),
            tooltip=[
                *([alt.Tooltip("count:Q", title="Count", aggregate="min")] if dropdown_verb_order.value == "count" else []),
                *([alt.Tooltip("count_is_verb:Q", title="Count Verb", aggregate="min")] if dropdown_verb_order.value == "count_verb" else []),
                *([alt.Tooltip("count_causative:Q", title="Count Causative", aggregate="min")] if dropdown_verb_order.value == "count_causative" else []),
            ],
        )
    ).interactive()

    mo.vstack([
       dropdown_step,
       dropdown_verb_order,
       chart,
    ])
    return


@app.cell
def _():
    verb_list_good = """appear
    chortle
    chuckle
    descend
    frown
    giggle
    glow
    grin
    groan
    laugh
    rise
    sing
    sleep
    smile
    vanish
    cry""".split("\n")

    verb_list_better = """laugh
    cry
    sing
    smile
    sleep""".split("\n")
    return verb_list_better, verb_list_good


@app.cell
def _(mo):
    select_filter_good_verbs = mo.ui.dropdown(["All Verbs", "Good Verbs", "Better Verbs"], value="All Verbs", label="Filter Verbs") # pyright: ignore
    checkbox_filter_verb_epoch_0 = mo.ui.checkbox(True, label="Filter Epoch 0") # pyright: ignore
    return checkbox_filter_verb_epoch_0, select_filter_good_verbs


@app.cell
def _(
    checkbox_filter_verb_epoch_0,
    dataset_ppl,
    dataset_verb_stat,
    list_step,
    mo,
    pl,
    select_filter_good_verbs,
    stats,
    verb_list_better,
    verb_list_good,
):
    def ttest(struct_ppl):
        arr_ppl = struct_ppl.to_arrow()
        ppl_causative, ppl_transitive = arr_ppl.field("ppl_causative").to_numpy(), arr_ppl.field("ppl_transitive").to_numpy()
        t_stat, p_value = stats.ttest_rel(ppl_causative, ppl_transitive, alternative="less")
        return p_value

    dataset_ttest = dataset_ppl.group_by("step", "verb").agg(
        (p_value := pl.struct("ppl_causative", "ppl_transitive").map_batches(ttest, pl.Float64, returns_scalar=True).alias("p_value")),
        pl.when(p_value < 0.05).then(pl.lit("*")).otherwise(pl.lit("")).alias("significant"),
        pl.col("ppl_causative").lt(pl.col("ppl_transitive")).sum().truediv(pl.len()).alias("hard_accuracy"),
    ).join(
        dataset_verb_stat,
        on="verb",
        how="inner"
    ).sort("verb").with_columns(
        pl.col("count_causative").fill_null(0).truediv(pl.col("count_is_verb").fill_null(1)).alias("proportion_causative"),
    )

    if select_filter_good_verbs.value == "Good Verbs":
        dataset_ttest = dataset_ttest.filter(pl.col("verb").is_in(verb_list_good))
    elif select_filter_good_verbs.value == "Better Verbs":
        dataset_ttest = dataset_ttest.filter(pl.col("verb").is_in(verb_list_better))

    if checkbox_filter_verb_epoch_0.value:
        dataset_ttest = dataset_ttest.filter(pl.col("step") == list_step[0])


    mo.vstack([
        mo.md(r"""
    ### Paired T-Test Results between Causative and Transitive Perplexities
    H1: $\mathrm{ppl}(\text{causative}) < \mathrm{ppl}(\text{transitive})$
    """ + f"""
    {select_filter_good_verbs}  
    {checkbox_filter_verb_epoch_0}
    """),
        mo.ui.dataframe(dataset_ttest, format_mapping={"p_value": "{:.4e}", "proportion_causative": "{:.4f}", "hard_accuracy": "{:.3%}"}),
    ])
    return


if __name__ == "__main__":
    app.run()
