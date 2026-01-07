# /// script
# [tool.marimo]
# ///

# %%
import marimo as mo
import polars as pl
import altair as alt
import numpy as np

# %%
dataset = pl.read_parquet("./var/scamp/scamp-plausible-llm-infer-grammatical-only.parquet")

dataset_full = pl.read_parquet("./var/scamp/scamp-plausible.parquet")

dataset = dataset_full.join(dataset.select(
    pl.all().exclude("sentence_raw", "tokens")
), on=["file_name", "file_line_idx", "is_grammatical"], coalesce=True, how="left")

# %%
dataset_gram = dataset.filter("is_grammatical").unnest("llm_infer")

# %%
# _df = dataset_gram.filter(pl.col("verdict").is_not_null()).group_by("file_name", "verdict").agg(
#     pl.len().alias("count")
# ).sort(["file_name", "verdict"])

# alt.Chart(_df).mark_bar().encode(
#     x=alt.X("file_name:N", title="Group"),
#     y=alt.Y("count:Q", title="Count"),
#     color=alt.Color("verdict:N", title="Verdict", sort=[True, False]),
#     tooltip=["file_name", "verdict", "count"]
# )

# %%
dataset_sample = (
    dataset_gram
    .filter(pl.col("verdict").is_not_null())
    .group_by("file_name", "verdict")
    .agg(
        pl.struct(pl.all().exclude(["file_name", "verdict"])).sample(
            pl.when(pl.len() > 2).then(2).otherwise(pl.len()),
            seed=42,
        ).alias("samples"),
        pl.len().alias("count"),
    )
    .explode("samples")
    .unnest("samples")
    .sort(["file_name", "verdict", "file_line_idx"], descending=[False, True, False])
    .with_row_index("sample_idx", offset=1)
    .select("sample_idx", "file_name", "file_line_idx", "verdict", pl.col("count").alias("total_count"), "sentence_raw", "output_reasoning")
)

# dataset_sample.select(pl.all().exclude("output_reasoning")).write_csv("scamp_con_samples.csv")
# dataset_sample.write_parquet("scamp_con_samples.parquet")

# dataset_sample = pl.read_parquet("scamp_con_samples.parquet")
# selector = mo.ui.number(start=1, stop=dataset_sample.height, step=1, label="Sample Index")

# # %%
# row_idx: int = selector.value - 1 # type: ignore
# sentence = dataset_sample[row_idx, "sentence_raw"]

# mo.vstack([
#     selector,
#     mo.md(f"`{dataset_sample[row_idx, 'file_name']}({dataset_sample[row_idx, 'total_count'] / 1000:.2%})::{dataset_sample[row_idx, 'file_line_idx']}`  - **{dataset_sample[row_idx, 'verdict']}**"),
#     mo.md(f"> {sentence}"),
#     mo.md(f"\n{dataset_sample[row_idx, 'output_reasoning']}")
# ])

# # %%
# dataset_verified = pl.read_csv("/home/tomyxw/Projects/2025/verb-overgen/verb-overgen/var/scamp-con-sample-verify.csv")

# # %%
# dataset_verified.select(
#     pl.col("verdict").eq(pl.col("manual")).filter(pl.col("verdict")).pipe(lambda tp: (
#         (tp.sum() / pl.col("verdict").sum()).alias("precision"),
#         (tp.sum() / pl.col("manual").sum()).alias("recall"),
#     ))
# )

# # %%
# seed_gen = np.random.default_rng(42)

# # %%


# # %%
# dataset_gram.sample(25, seed=seed_gen.integers(np.iinfo(np.uint64).min, np.iinfo(np.uint64).max, size=1, dtype=np.uint64)[0]).select(pl.col("verdict").value_counts()).unnest(pl.selectors.all()).sort("verdict", descending=True)
