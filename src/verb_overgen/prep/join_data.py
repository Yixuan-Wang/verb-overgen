from os import environ
import polars as pl

def read_lines_into_polars(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return pl.DataFrame({
        "text": lines
    })

DATASET_BASE = environ["DATA_PATH_BASE"]

DATASET = {
    "train": read_lines_into_polars(f"{DATASET_BASE}/babylm/babylm_10M/aochildes.train"),
    "dev": read_lines_into_polars(f"{DATASET_BASE}/babylm/babylm_dev/aochildes.dev"),
    "test": read_lines_into_polars(f"{DATASET_BASE}/babylm/babylm_test/aochildes.test"),
}

# Join the datasets
dataset = pl.concat([df.with_columns(split=pl.lit(split)) for split, df in DATASET.items()])
dataset = dataset.with_columns(split=dataset["split"].cast(pl.Enum(["train", "dev", "test"])))

path = f"{DATASET_BASE}/babylm/aochildes.parquet"
if input("Save file? (y/n): ") == "y":
    dataset.write_parquet(path)
    print("Saved.")
