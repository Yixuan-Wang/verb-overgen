import polars as pl

dataset = pl.read_parquet("./data/exp/overgen_trajectory-260410-260410T2356.parquet")

index_dataset = [col for col in dataset.columns if not col.startswith("eval") and not col.startswith("train")]
dataset_eval = dataset.lazy().unpivot(index=index_dataset, on=pl.selectors.starts_with("eval"), variable_name="metric", value_name="loss").with_columns(
    pl.col("metric").str.strip_prefix("eval/loss_mean/").str.split("/")
).with_columns(
    pl.col("metric").list.get(0).alias("verb"),
    pl.col("metric").list.get(1).alias("sentence_type"),
    pl.col("run_subset").replace_strict({
        "remove_preemption": "-P",
        "remove_entrenchment": "+P",
    })
).drop("metric").collect()


dataset_verb_stats = pl.read_parquet("./var/results/verb_stats.parquet")
dataset_final_loss = (
        dataset_eval.group_by(
            pl.col("verb"),
            pl.col("run_subset"),
            pl.col("run_seed"),
            pl.col("sentence_type"),
        )
        .agg(
            pl.col("loss")
            .sort_by("step")
            .last()
        )
    )

dataset_final_loss_corr = dataset_final_loss.join(
    dataset_verb_stats, on="verb"
).sort(["verb", "run_subset", "run_seed", "sentence_type"])
