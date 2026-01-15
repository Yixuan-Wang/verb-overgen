from typing import TypedDict, cast

import polars as pl
from torch.utils.data import Dataset as TorchDataset

from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data

class DataOneTestset(DataOne, TypedDict):
    index_data: int

class DataBatchTestset(DataBatch, TypedDict):
    index_data: list[int]

class DatasetTestset(TorchDataset[DataOneTestset]):
    def __init__(self, *, input_path: str, token_id_pad: int) -> None:
        self.dataframe = self._load_data(input_path)
        self.token_id_pad = token_id_pad
        
    def _load_data(self, input_path) -> pl.DataFrame:
        dataset = pl.read_csv(input_path).with_row_index("index_data")
        return dataset

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> DataOneTestset:
        input_ids = self.dataframe.item(index, "input_ids")
        data = cast(DataOneTestset, get_one_data(input_ids))
        data["index_data"] = index
        return data

    def collate_fn(self, batch: list[DataOneTestset]) -> DataBatchTestset:
        data = collate(batch, token_id_pad=self.token_id_pad)
        data = cast(DataBatchTestset, data)
        data["index_data"] = [item["index_data"] for item in batch]
        return data
