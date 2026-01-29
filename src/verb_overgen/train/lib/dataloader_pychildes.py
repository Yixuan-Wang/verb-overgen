from __future__ import annotations

import posix
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import TypedDict, cast

import ipdb
import polars as pl
import torch
from apfel import imperative
from loguru import logger
from torch.utils.data import Dataset, Sampler

from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, collate_pack_pad, get_one_data


class DataBatchPyChildes(DataBatch, TypedDict):
    index: list[int]
    id_file_utterance: list[str]


class DataOnePyChildes(DataOne, TypedDict):
    index: int
    id_file_utterance: str


class DatasetPyChildes(Dataset[DataOnePyChildes]):
    dataframe: pl.DataFrame
    token_id_pad: int
    max_len: int
    expr_keep: pl.Expr | Sequence[pl.Expr] | None

    def __init__(
        self,
        *,
        input_path: Path,
        token_id_pad: int,
        max_len: int,
        max_token_per_batch: int,
        expr_keep: pl.Expr | Sequence[pl.Expr] | None = None,
    ):
        """
        Dataset for PyChildes data stored in a Parquet file.

        Args:
            input_path (Path): Path to the Parquet file containing the dataset.
            token_id_pad (int): Token ID used for padding sequences.
            max_len (int): Maximum sequence length allowed.
            max_token_per_batch (int): Maximum number of tokens allowed per batch during sampling.
            expr_keep (pl.Expr | Sequence[pl.Expr] | None): Optional Polars expression(s) to filter data during collation. The expressions should evaluate to a boolean value, where True indicates the data should be included, and the expressions are combined using a logical AND operation. If None, no filtering is applied.
        """
        self.dataframe = self._load_data(input_path)
        self.token_id_pad = token_id_pad
        self.max_len = max_len
        # self.expr_keep = expr_keep
        if expr_keep is not None:
            raise NotImplementedError("Filtering under packing is not yet implemented.")
        # if self.expr_keep is not None:
        #     self.collate_fn = self.collate_fn_filter
        # else:
        #     self.collate_fn = self.collate_fn_no_filter

    def _load_data(self, input_path: Path):
        df = pl.read_parquet(input_path)
        if df.schema.get("input_ids") != pl.List(pl.Int32):
            raise KeyError(
                "Pre-tokenize dataset first using `train.run_tokenize`. The `input_ids` column should be of type `List(Int32)`."
            )

        if "index" not in df.columns:
            df = df.select(
                pl.concat_str(pl.col("id_file"), pl.lit(":"), pl.col("id_utterance")).alias("id_file_utterance"),
                pl.all(),
            )

        # Note: The filter and preprocess function are moved to `train.run_tokenize.load_data`.
        df = df.with_columns(pl.col("input_ids").cast(pl.List(pl.Int64))).with_row_index("index")

        if df.select(pl.col("input_ids").list.len().max()).item() > self.max_len:
            raise ValueError(f"Maximum sequence length in dataset exceeds max_len={self.max_len}")

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
        data["id_file_utterance"] = self.dataframe.item(index, "id_file_utterance")
        return data

    def create_sampler(self, *, shuffle: bool | torch.Generator | None = None) -> SamplerBin:
        """Create a Sampler that bins samples based on their lengths.

        Args:
            shuffle (bool | torch.Generator | None): If True, shuffle the bins. If a torch.Generator is provided, use it for shuffling. If False or None, do not shuffle.
        """
        list_len = cast(Sequence[int], self.dataframe.select(pl.col("input_ids").list.len()).to_series())
        return SamplerBin(list_len, shuffle=shuffle, max_token_per_batch=self.max_token_per_batch)

    def collate_fn(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        data: DataBatchPyChildes = collate_pack_pad(batch, token_id_pad=self.token_id_pad, max_len=self.max_len)  # type: ignore
        data["index"] = [item["index"] for item in batch]
        data["id_file_utterance"] = [item["id_file_utterance"] for item in batch]
        return data

    # def collate_fn_filter(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
    #     list_index_original = [item["index"] for item in batch]

    #     list_ok: list[bool] = [
    #         (self.dataframe.slice(i, 1).select(self.expr_keep).select(pl.all_horizontal(pl.all())).item())
    #         for i in list_index_original
    #     ]
    #     filtered_batch = [item for ok, item in zip(list_ok, batch) if ok]

    #     data: DataBatchPyChildes = collate(filtered_batch, token_id_pad=self.token_id_pad)  # type: ignore
    #     data["index"] = [item["index"] for item in filtered_batch]
    #     data["id_file_utterance"] = [item["id_file_utterance"] for item in filtered_batch]
    #     return data

class SamplerBin(Sampler[list[int]]):
    bin_length: list[tuple[int, int]]

    def __init__(self, list_len: Sequence[int], *, max_token_per_batch: int, shuffle: bool | torch.Generator | None = None):
        self.bin_length = list(self._create_bin_length(list_len, max_token_per_batch=max_token_per_batch))

        if shuffle:
            generator = torch.Generator() if shuffle is True else shuffle
            indices = torch.randperm(len(self.bin_length), generator=generator).tolist()
            self.bin_length = [self.bin_length[i] for i in indices]

    @staticmethod
    def _create_bin_length(list_len: Sequence[int], *, max_token_per_batch: int) -> Generator[tuple[int, int], None, None]:
        """The naive greedy binning algorithm.

        Args:
            list_len (Sequence[int]): A sequence of lengths to be binned.
            max_token_per_batch (int): The maximum allowed length for each bin.

        Yields:
            Generator[tuple[int, int], None, None]: Tuples of (start_index, end_index) for each bin.

        Raises:
            ValueError: If any single item length exceeds `max_token_per_batch`.
        """
        acc, acc_start = 0, 0
        total = len(list_len)

        for i in range(total):
            l = list_len[i]
            if l > max_token_per_batch:
                raise ValueError(f"Single item length {l} at index {i} exceeds max_len {max_token_per_batch}.")

        for i in range(total):
            l = list_len[i]
            if acc + l > max_token_per_batch:
                yield (acc_start, i)
                acc = 0
                acc_start = i
            acc += l

        if acc_start < total:
            yield (acc_start, total)

    def __iter__(self):
        return (list(range(start, end)) for start, end in self.bin_length)

    def __len__(self):
        return len(self.bin_length)


if __name__ == "__main__":
    from itertools import islice

    from torch.utils.data import DataLoader

    PATH_TRAIN = "./data/pychildes/pychildes-morphosyntax_postprocess.20260102_20251024.parquet"

    expr_keep = pl.col("features").list.contains(FEATURE_CAUSATIVE).not_()

    dataset = DatasetPyChildes(
        input_path=Path(PATH_TRAIN),
        token_id_pad=0,
        max_token_per_batch=4096,
        max_len=128,
        # expr_keep=expr_keep,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.create_sampler(shuffle=True),
        collate_fn=dataset.collate_fn,
    )
