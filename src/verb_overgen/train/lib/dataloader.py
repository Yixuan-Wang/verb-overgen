from collections.abc import Sequence
from typing import TypedDict

import torch
import polars as pl
from jaxtyping import Int
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class DataOne(TypedDict):
    input_ids: Int[Tensor, "seq"]


class DataBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    seq_length: Int[Tensor, "batch"]

def strip_last_token(input_ids: pl.Expr) -> pl.Expr:
    """Strip the last token from the input_ids."""
    return input_ids.list.slice(0, input_ids.list.len() - 1)    

def get_one_data(input_ids: Sequence[int]) -> DataOne:
    input_ids: Int[Tensor, "seq"] = torch.tensor(
        input_ids,
        dtype=torch.int64,
    )

    return {
        "input_ids": input_ids,
    }

def collate(batch: Sequence[DataOne], *, token_id_pad: int) -> DataBatch:
    """Collate a batch of data with padding."""
    seq_length = torch.tensor([item["input_ids"].size(0) for item in batch])

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=token_id_pad,
    )
    # attention_mask = pad_sequence(
    #     [item["attention_mask"] for item in batch],
    #     batch_first=True,
    #     padding_value=0,
    # )
    # position_ids = pad_sequence(
    #     [item["position_ids"] for item in batch],
    #     batch_first=True,
    #     padding_value=0,
    # )

    return {
        "input_ids": input_ids,
        # "attention_mask": attention_mask,
        # "position_ids": position_ids,
        "seq_length": seq_length,
    }

