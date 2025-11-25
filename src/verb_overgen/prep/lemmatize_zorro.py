import itertools
from pathlib import Path
from rich import print as rprint
import polars as pl
from tqdm import tqdm
import ray
import ray.data as ray_data
import pyarrow as pa

from verb_overgen.prep.lemmatize_data import text_to_lemma, lemmatize_dtype

if __name__ == "__main__":
    list_df = []
    for path_file in Path("/u501/y4466wan/projects/2025/verb-overgen/data/zorro/zorro/sentences/babyberta").iterdir():
        with open(path_file, "r") as f:
            ser_setence = pl.Series("sentence", f)
            df = pl.DataFrame({"sentence": ser_setence})
            df = df.with_row_index("file_line_idx", 1).with_columns(
                file_name = pl.lit(path_file.stem),
                is_grammatical = (pl.col("file_line_idx") % 2 == 0),
            ).select(
                "file_name",
                "file_line_idx",
                "is_grammatical",
                "sentence",
            )

            list_df.append(df)

    dataset: pl.DataFrame = pl.concat(list_df)

    def lemmatize(tbl: pa.Table) -> pa.Table:
        df: pl.DataFrame = pl.from_arrow(tbl) # pyright: ignore
        df = df.rename({"sentence": "sentence_raw"}).with_columns(
            pl.col("sentence_raw").map_elements(text_to_lemma, return_dtype=lemmatize_dtype).struct.unnest()
        )
        return df.to_arrow()
    
    ray.init(num_cpus=16, ignore_reinit_error=True)
    plan = ray_data.from_arrow(dataset.to_arrow(), override_num_blocks=16).map_batches(lemmatize, batch_size=4096, batch_format="pyarrow", num_cpus=1, compute=ray_data.TaskPoolStrategy(size=16))

    dataset_lemmatized = pl.from_arrow(ray.get(plan.to_arrow_refs())) # pyright: ignore
    dataset_lemmatized: pl.DataFrame
    dataset_lemmatized = dataset_lemmatized.sort(["file_name", "file_line_idx"])
    dataset_lemmatized.write_parquet("/u501/y4466wan/projects/2025/verb-overgen/data/zorro/zorro_babyberta_lemmatized.parquet")