from typing import TypedDict, cast

import einops as eo
import polars as pl
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn.utils.rnn import unpad_sequence
from torch.utils.data import Dataset as TorchDataset

from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data


class DataOneScamp(DataOne, TypedDict):
    index_data: int


class DataBatchScamp(DataBatch, TypedDict):
    index_data: list[int]


class DatasetScamp(TorchDataset[DataOneScamp]):
    def __init__(self, *, input_path: str, token_id_pad: int):
        self.dataframe = self._load_data(input_path)
        self.token_id_pad = token_id_pad

    def _load_data(self, input_path: str):
        dataset = pl.read_parquet(input_path).with_row_index("index_data")
        return dataset

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> DataOneScamp:
        input_ids = self.dataframe.item(index, "input_ids")
        data = cast(DataOneScamp, get_one_data(input_ids))
        data["index_data"] = self.dataframe.item(index, "index_data")
        return data

    def collate_fn(self, batch: list[DataOneScamp]) -> DataBatchScamp:
        data = cast(DataBatchScamp, collate(batch, token_id_pad=self.token_id_pad))
        data["index_data"] = [item["index_data"] for item in batch]
        return data


@torch.no_grad()
def evaluate_scamp(
    model: torch.nn.Module,
    batch: DataBatchScamp,
) -> list[Float[Tensor, "ragged_seq"]]:
    input_ids: Int[Tensor, "batch seq"] = batch["input_ids"]
    attention_mask: Int[Tensor, "batch seq"] = batch["attention_mask"]
    seq_length: Int[Tensor, "batch"] = batch["seq_length"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    prob: Float[Tensor, "batch seq vocab"] = outputs.logits.softmax(dim=-1)

    _this = torch.gather(
        prob[..., :-1, :],
        dim=-1,
        index=eo.rearrange(input_ids[..., 1:], "batch seq -> batch seq 1"),
    )
    token_prob: Float[Tensor, "batch seq"] = _this[..., 0]

    # Truncate each tensor to its original length and return as a list
    token_prob = token_prob.cpu()
    seq_length_without_bos: Int[Tensor, "batch"] = (seq_length - 1).clamp(min=0).cpu()
    list_token_prob = unpad_sequence(token_prob, lengths=seq_length_without_bos, batch_first=True)

    return list_token_prob
