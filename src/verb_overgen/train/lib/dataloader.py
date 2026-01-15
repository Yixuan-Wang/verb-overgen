from collections.abc import Sequence
from typing import TypedDict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from jaxtyping import Int

class DataOne(TypedDict): 
    input_ids: Int[Tensor, "seq"]
    attention_mask: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]

class DataBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    attention_mask: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    seq_length: Int[Tensor, "batch"]


def get_one_data(input_ids: Sequence[int]) -> DataOne:
    # Here the `input_ids` are already patched with BOS and EOS tokens.
    input_ids: Int[Tensor, "seq"] = torch.tensor(
        input_ids,
        dtype=torch.int64,
    )
    seq_len = input_ids.size(0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(seq_len),
    }

def collate(batch: Sequence[DataOne], *, token_id_pad: int) -> DataBatch:
    seq_length = torch.tensor([item["input_ids"].size(0) for item in batch])

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=token_id_pad,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    position_ids = pad_sequence(
        [item["position_ids"] for item in batch],
        batch_first=True,
        padding_value=-1,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "seq_length": seq_length,
    }
