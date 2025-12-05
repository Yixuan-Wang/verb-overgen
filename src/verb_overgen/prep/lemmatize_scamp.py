import itertools
from pathlib import Path
from rich import print as rprint
import polars as pl
from tqdm import tqdm
import ray
import ray.data as ray_data
import pyarrow as pa

from verb_overgen.prep.lemmatize_data import text_to_spacy, spacy_metadata_dtype

def create_scamp_dataset(dir_scamp: Path) -> pl.DataFrame:
    def _load_scamp():
        for path_file in dir_scamp.glob("*.tsv"):
            with path_file.open("r") as f:
                lines = [line.strip().split("\t") for line in f.readlines()]

            yield pl.DataFrame(lines, schema=pl.Schema({
                "sentence_raw_grammatical": pl.String,
                "sentence_raw_ungrammatical": pl.String,
            }), orient="row").with_columns(
                pl.lit(path_file.stem).alias("file_name"),
            ).with_row_index("file_line_idx", 1).select([
                "file_name",
                "file_line_idx",
                "sentence_raw_grammatical",
                "sentence_raw_ungrammatical",
            ])

    scamp = pl.concat(_load_scamp(), how="vertical")

    scamp = scamp.unpivot(index=["file_name", "file_line_idx"], value_name="sentence_raw", variable_name="grammaticality").select(
        pl.col("file_name", "file_line_idx",),
        pl.col("grammaticality").eq("sentence_raw_grammatical").alias("is_grammatical"),
        pl.col("sentence_raw"),
    ).sort(["file_name", "file_line_idx", "is_grammatical"], descending=[False, False, True])

    return scamp




if __name__ == "__main__":
    dir_scamp = Path("/u501/y4466wan/projects/2025/verb-overgen/data/scamp/scamp-plausible/")
    dataset: pl.DataFrame = create_scamp_dataset(dir_scamp)

    def lemmatize(tbl: pa.Table) -> pa.Table:
        df: pl.DataFrame = pl.from_arrow(tbl) # pyright: ignore
        df = df.with_columns(
            pl.col("sentence_raw").map_elements(text_to_spacy, spacy_metadata_dtype).struct.unnest()
        )
        return df.to_arrow()
    
    # ray.init(num_cpus=16, ignore_reinit_error=True)
    # plan = ray_data.from_arrow(dataset.to_arrow()).map_batches(lemmatize, batch_size=4096, batch_format="pyarrow", num_cpus=1, compute=ray_data.TaskPoolStrategy(size=16))

    # dataset_with_spacy = pl.from_arrow(ray.get(plan.to_arrow_refs())) # pyright: ignore
    dataset_with_spacy: pl.DataFrame
    dataset_with_spacy = dataset.with_columns(
        pl.col("sentence_raw").map_elements(text_to_spacy, spacy_metadata_dtype).struct.unnest()
    )
    # dataset_with_spacy = dataset_with_spacy.sort(["file_name", "file_line_idx", "is_grammatical"], descending=[False, False, True])
    dataset_with_spacy.write_parquet("/u501/y4466wan/projects/2025/verb-overgen/data/scamp/scamp-plausible_spacy.parquet")