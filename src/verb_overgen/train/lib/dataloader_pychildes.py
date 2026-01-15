from pathlib import Path
from typing import TypedDict

import polars as pl
from torch.utils.data import Dataset

from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data


class DataBatchPyChildes(DataBatch, TypedDict): ...


class DataOnePyChildes(DataOne, TypedDict): ...


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

        # Note: The filter and preprocess function are moved to `train.run_tokenize.load_data`.
        df = df.with_columns(pl.col("input_ids").cast(pl.List(pl.Int64)))
        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> DataOnePyChildes:
        input_ids = self.dataframe.item(index, "input_ids")
        return get_one_data(input_ids)

    def collate_fn(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        return collate(batch, token_id_pad=self.token_id_pad)
