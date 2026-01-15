from __future__ import annotations
from collections.abc import Iterator
import itertools
from typing import Literal, Self, TypedDict
import weakref
import numpy as np
import polars as pl
import os
from apfel.container.once import Lazy
import torch
import transformers as hf

from verb_overgen.util.pl_util import batch
from dataclasses import dataclass, field


def split(split: Literal["train", "dev", "test"]) -> pl.Expr:
    """
    Get the split of the dataset.
    """
    return pl.col("split") == split

@dataclass(kw_only=True)
class Dataloader:
    """
    The dataloader for training.
    Do not call the constructor directly. Use `Dataloader.from_dataframe` instead.
    """

    dataset_src: pl.DataFrame
    """
    The dataframe holding the dataset source.
    
    Cols:
        index(UInt64): The index of the row.
        split(str): The split of the dataset. Should be the same for all rows.
        tokenized(Object): The tokenized input, stored as `hf`'s `BatchEncoding`.
        tokenized_len(UInt64): The length of the tokenized input.
    """

    bos_token: int
    """
    The <BOS> token to be prepended to the input.
    """

    # pad_token: int
    # """
    # The <PAD> token to be used for padding.
    # """

    batch_size: int
    """
    The batch size for the dataloader when generating batches.
    """

    seed_global_shuffle: Iterator[int] | Iterator[np.uint32]
    """
    The seed generator for shuffling the dataset globally, before batching.
    """

    seed_batch_shuffle: Iterator[int] | Iterator[np.uint32]
    """
    The seed generator for shuffling the batches.
    """

    @classmethod
    def from_dataframe(
        cls,
        dataset: pl.DataFrame,
        *,
        batch_size: int,
        bos_token: int,
        seed_global_shuffle: Iterator[int] | Iterator[np.uint32] | int,
        seed_batch_shuffle: Iterator[int] | Iterator[np.uint32] | int,
    ) -> Self:
        """
        Create a dataloader from a dataframe containing multiple splits.

        Args:
            dataset (pl.DataFrame): The dataframe containing the dataset.
            batch_size (int): The batch size for the dataloader when generating batches.
            bos_token (int): The <BOS> token to be prepended to the input.
            seed_global_shuffle (int | Iterator[int]): The seed generator for shuffling the dataset globally, before batching.
            seed_batch_shuffle (int | Iterator[int]): The seed generator for shuffling the batches.
        """

        dataset_src = (
            dataset
            .select(pl.all().exclude("index"))
            .with_row_index("index")
            .with_columns(
                tokenized_len=pl.col("tokenized").map_elements(
                    lambda x: len(x.input_ids),
                    pl.UInt64,
                ),
            )
        )

        seed_global_shuffle = itertools.repeat(seed_global_shuffle) if isinstance(seed_global_shuffle, int) else seed_global_shuffle
        seed_batch_shuffle = itertools.repeat(seed_batch_shuffle) if isinstance(seed_batch_shuffle, int) else seed_batch_shuffle

        return cls(
            dataset_src=dataset_src,
            bos_token=bos_token,
            batch_size=batch_size,
            seed_global_shuffle=seed_global_shuffle,
            seed_batch_shuffle=seed_batch_shuffle,
        )

    def yield_plan(self) -> pl.DataFrame:
        """
        Yield a batching plan for the dataloader, advancing the internal seed generators.
        """

        plan = (
            self.dataset_src.sample(
                fraction=1,
                shuffle=True,
                seed=int(next(self.seed_global_shuffle)),
            )
            .sort(
                "tokenized_len",
                maintain_order=True,
            )
            .select(
                index_src=pl.col("index").pipe(batch, self.batch_size),
            )
            .sample(
                fraction=1,
                shuffle=True,
                seed=int(next(self.seed_batch_shuffle)),
            )
        )

        return plan

    def batch_count(self) -> int:
        """
        Return the number of batches in the dataloader.
        """
        if self.dataset_src.height % self.batch_size == 0:
            return self.dataset_src.height // self.batch_size
        else:
            return self.dataset_src.height // self.batch_size + 1

    def __len__(self) -> int:
        """
        Return the number of data points in the dataloader, before batching.
        """
        return self.dataset_src.height

    def __iter__(self) -> DataloaderIter:
        """
        Return an iterator for the dataloader.
        Notice that each call to `__iter__` will mutate the internal state of the dataloader.
        """
        return DataloaderIter(
            dataloader=weakref.ref(self),
            batch_plan=self.yield_plan(),
        )

@dataclass(kw_only=True)
class Input:
    input_ids: torch.Tensor
    target: torch.Tensor
    batch_index: int
    batch_src: pl.DataFrame


@dataclass(kw_only=True)
class DataloaderIter:
    """
    The stateful iterator for the dataloader, created by `Dataloader.__iter__`.

    Should not be used directly.
    """

    dataloader: weakref.ref[Dataloader]
    batch_plan: pl.DataFrame

    cur: int = field(init=False, default=0)
    """
    The cursor for the iterator.
    """

    batch_count_total: int = field(init=False)

    def __post_init__(self) -> None:
        dataloader = self.dataloader()
        if dataloader is None:
            raise ValueError("Dataloader is not set.")

        self.batch_count_total = self.batch_plan.height

    def __len__(self) -> int:
        """
        Return the number of batches in the iterator.
        """
        return self.batch_count_total

    def reset(self) -> None:
        """
        Reset the iterator to the beginning.
        """
        self.cur = 0

    def __iter__(self) -> Self:
        """
        Return the iterator itself.
        """
        return self

    def __next__(self) -> Input:
        """
        Get the next batch of data.
        """
        dataloader = self.dataloader()
        if dataloader is None:
            raise ValueError("Dataloader is not set.")

        if self.cur >= self.batch_count_total:
            raise StopIteration

        c = self.cur
        self.cur += 1

        b: pl.Series = self.batch_plan[c, "index_src"]

        batch = dataloader.dataset_src[b]

        # Do truncation according to the minimum length of the batch
        seq_length: int = batch.select(pl.col("tokenized_len").min()).item()

        bos_token = dataloader.bos_token
        tokens = batch.select(
            pl.col("tokenized")
            .map_elements(
                lambda x, l=seq_length, b=bos_token: [b, *x.input_ids[:l]],
                pl.List(pl.Int32),
            )
            .list.to_array(seq_length + 1)
        ).to_series().to_numpy()

        input_ids = torch.tensor(tokens[:, :-1])
        target = torch.tensor(tokens[:, 1:], device=torch.device("cpu"), dtype=torch.int64)

        return Input(
            input_ids=input_ids,
            target=target,
            batch_src=batch,
            batch_index=c,
        )