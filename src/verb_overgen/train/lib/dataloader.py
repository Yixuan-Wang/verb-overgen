from collections.abc import Sequence
from typing import TypedDict

import torch
from jaxtyping import Int
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class DataOne(TypedDict):
    input_ids: Int[Tensor, "seq"]
    attention_mask: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]


class DataBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    attention_mask: Int[Tensor, "batch seq"] | Int[Tensor, "batch seq seq"]
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
    """Collate a batch of data with padding."""
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
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "seq_length": seq_length,
    }


def collate_pack_pad(batch: Sequence[DataOne], *, max_len: int, token_id_pad: int) -> DataBatch:
    """Collate a batch of data by packing and padding sequences to a maximum length.
    No truncation or cutting is performed; sequences exceeding `max_len` will raise an error.

    Args:
        batch (Sequence[DataOne]): A sequence of DataOne items.
        max_len: The maximum length to pack the sequences to.
        token_id_pad: The token ID used for padding.

    Returns:
        DataBatch: A collated batch of data with sequences packed and padded to `max_len`.

    Raises:
        ValueError: If any sequence in the batch exceeds `max_len`.
    """

    list_seq_length_flat = [item["input_ids"].size(0) for item in batch]
    seq_length_flat = torch.tensor(list_seq_length_flat)

    _acc, _start = 0, 0
    list_ranges: list[tuple[int, int]] = []
    for _idx, _size in enumerate(list_seq_length_flat):
        if _size > max_len:
            raise ValueError(f"Sequence at index {_idx} exceeds max_len {max_len} with size {_size}.")
        if _acc + _size > max_len:
            list_ranges.append((_start, _idx))
            _acc, _start = 0, _idx
        _acc += _size

    if _start < (_len := len(batch)):
        list_ranges.append((_start, _len))

    seq_length: Int[Tensor, "batch"] = torch.tensor(
        [seq_length_flat[start:end].sum() for start, end in list_ranges]
    )

    doc_ids: Int[Tensor, "batch seq"] = pad_sequence(
        [torch.cat([
            torch.full((list_seq_length_flat[_idx_flat],), _idx_local)
            for _idx_local, _idx_flat in enumerate(range(_start, _end))
        ]) for _start, _end in list_ranges],
        batch_first=True,
        padding_value=-1,
    )

    input_ids: Int[Tensor, "batch seq"] = pad_sequence(
        [torch.cat([batch[_idx_flat]["input_ids"] for _idx_flat in range(_start, _end)], dim=0) for _start, _end in list_ranges],
        batch_first=True,
        padding_value=token_id_pad,
    )

    # attention_mask: Int[Tensor, "batch 1 seq seq"]
    # attention_mask_not_padding = (_not_padding := input_ids != token_id_pad).unsqueeze(-1) & _not_padding.unsqueeze(-2)
    # if conversational_mask:
    #     attention_mask = torch.tril(attention_mask_not_padding).int().unsqueeze(1)
    # else:
    #     attention_mask_same_sentence = (doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2))
    #     attention_mask = torch.tril(attention_mask_same_sentence & attention_mask_not_padding).int().unsqueeze(1)
    attention_mask = input_ids.ne(token_id_pad).int()

    position_ids: Int[Tensor, "batch seq"] = pad_sequence(
        [torch.cat([batch[_idx_flat]["position_ids"] for _idx_flat in range(_start, _end)], dim=0) for _start, _end in list_ranges],
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "seq_length": seq_length,
    }
