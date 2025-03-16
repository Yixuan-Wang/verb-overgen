from pathlib import Path
import fire
import polars as pl

from verb_overgen.util.log_util import report_progress, register_workload_on_df
from verb_overgen.prep.lib.lexical import vec_to_lemma

lemmatize_dtype = pl.Struct({
    "lemma": pl.List(pl.String),
    "pos": pl.List(pl.String),
})

def main(path: str):
    dataset = pl.read_parquet(path)

    dataset_nontrivial = dataset.filter(pl.col("parse").is_not_null())

    dataset_lemma = dataset_nontrivial.pipe(register_workload_on_df).select(
        pl.col("index"),
        pl.col("parse").struct.field("word").map_elements(report_progress(vec_to_lemma), lemmatize_dtype).struct.unnest(),
    )

    # Notice that this join is very slow.
    dataset = dataset.join(dataset_lemma, on="index", how="left", maintain_order="left")
    new_path = Path(path).with_suffix(".{timestamp}.lemma.parquet".format(
        timestamp=__import__("datetime").datetime.now().strftime(r"%Y%m%dT%H%M%S")
    ))
    dataset.write_parquet(new_path)

if __name__ == "__main__":
    fire.Fire(main)
