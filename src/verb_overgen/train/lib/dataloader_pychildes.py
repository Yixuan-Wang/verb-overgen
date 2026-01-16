from pathlib import Path
from typing import TypedDict

import polars as pl
from torch.utils.data import Dataset
from loguru import logger

from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data


class DataBatchPyChildes(DataBatch, TypedDict):
    index: list[str]


class DataOnePyChildes(DataOne, TypedDict):
    index: str


class DatasetPyChildes(Dataset[DataOnePyChildes]):
    def __init__(
        self,
        *,
        input_path: Path,
        token_id_pad: int,
    ):
        self.dataframe = self._load_data(input_path)
        self.token_id_pad = token_id_pad

    def _load_data(self, input_path: Path):
        df = pl.read_parquet(input_path)
        if df.schema.get("input_ids") != pl.List(pl.Int32):
            raise KeyError(
                "Pre-tokenize dataset first using `train.run_tokenize`. The `input_ids` column should be of type `List(Int32)`."
            )

        if "index" not in df.columns:
            df = df.select(
                pl.concat_str(pl.col("id_file"), pl.lit(":"), pl.col("id_utterance")).alias("index"),
                pl.all(),
            )

        # Note: The filter and preprocess function are moved to `train.run_tokenize.load_data`.
        df = df.with_columns(pl.col("input_ids").cast(pl.List(pl.Int64)))
        return df

    def debug_truncate(self, n_samples: int):
        logger.info(f"Truncating training set to {n_samples} samples for debugging.")
        self.dataframe = self.dataframe.head(n_samples)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> DataOnePyChildes:
        input_ids = self.dataframe.item(index, "input_ids")
        data: DataOnePyChildes = get_one_data(input_ids)  # type: ignore
        data["index"] = self.dataframe.item(index, "index")
        return data

    def collate_fn(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        data: DataBatchPyChildes = collate(batch, token_id_pad=self.token_id_pad) # type: ignore
        data["index"] = [item["index"] for item in batch]
        return data
