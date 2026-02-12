from __future__ import annotations

from collections import Counter
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TypedDict, cast

import polars as pl
import torch
from torch import Tensor
from loguru import logger
import tokenizers as hftk
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as ptl
from jaxtyping import Int

from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.train.lib.dataloader import DataBatch, DataOne, get_one_data, strip_last_token


class DataBatchPyChildes(DataBatch, TypedDict):
    index: list[int]
    id_file_utterance: list[str]
    attention_mask: Int[Tensor, "batch seq"]


class DataOnePyChildes(DataOne, TypedDict):
    index: int
    id_file: list[str]
    id_file_utterance: str


class DatasetPyChildes(Dataset[DataOnePyChildes]):
    dataframe: pl.DataFrame
    token_id_pad: int
    max_len: int
    max_token_per_batch: int
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
        self.token_id_pad = token_id_pad
        self.max_len = max_len
        self.max_token_per_batch = max_token_per_batch
        # self.expr_keep = expr_keep
        if expr_keep is not None:
            raise NotImplementedError("Filtering under packing is not yet implemented.")
        # if self.expr_keep is not None:
        #     self.collate_fn = self.collate_fn_filter
        # else:
        #     self.collate_fn = self.collate_fn_no_filter
        self.dataframe = self._load_data(input_path)

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
        df = df.with_columns(
            pl.col("input_ids").pipe(strip_last_token).cast(pl.List(pl.Int64))
        ).with_row_index("index")

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
        data["id_file"] = self.dataframe.item(index, "id_file")
        data["id_file_utterance"] = self.dataframe.item(index, "id_file_utterance")
        return data

    def create_sampler(self, *, shuffle: bool | torch.Generator | None = None) -> SamplerBin:
        """Create a Sampler that bins samples based on their lengths.

        Args:
            shuffle (bool | torch.Generator | None): If True, shuffle the bins. If a torch.Generator is provided, use it for shuffling. If False or None, do not shuffle.
        """
        list_len = cast(Sequence[int], self.dataframe.select(pl.col("input_ids").list.len()).to_series())
        return SamplerBin(list_len, shuffle=shuffle, max_token_per_batch=self.max_token_per_batch)
        
    @staticmethod
    def collate_utterances(batch: Sequence[DataOnePyChildes], *, max_len: int, token_id_pad: int) -> DataBatchPyChildes:
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
        list_id_file_flat = [item["id_file"] for item in batch]
        seq_length_flat = torch.tensor(list_seq_length_flat)
    
        _acc, _start = 0, 0
        _prev_file: str | None = None
        list_ranges: list[tuple[int, int]] = []
        for _idx, (_size, _id_file) in enumerate(zip(list_seq_length_flat, list_id_file_flat, strict=True)):
            if _prev_file is None:
                _prev_file = _id_file

            if _size > max_len:
                raise ValueError(f"Sequence at index {_idx} exceeds max_len {max_len} with size {_size}.")
            if (
                _acc + _size > max_len
                or _id_file != _prev_file
            ):
                list_ranges.append((_start, _idx))
                _acc, _start = 0, _idx
                _prev_file = _id_file
            _acc += _size
    
        if _start < (_len := len(batch)):
            list_ranges.append((_start, _len))
    
        seq_length: Int[Tensor, "batch"] = torch.tensor(
            [seq_length_flat[start:end].sum() for start, end in list_ranges]
        )
    
        # doc_ids: Int[Tensor, "batch seq"] = pad_sequence(
        #     [torch.cat([
        #         torch.full((list_seq_length_flat[_idx_flat],), _idx_local)
        #         for _idx_local, _idx_flat in enumerate(range(_start, _end))
        #     ]) for _start, _end in list_ranges],
        #     batch_first=True,
        #     padding_value=-1,
        # )
    
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
    
        # position_ids: Int[Tensor, "batch seq"] = torch.arange(input_ids.size(1)).unsqueeze(0).expand(input_ids.size(0), -1)

        id_file_utterance = [item["id_file_utterance"] for item in batch]
        index = [item["index"] for item in batch]
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_length": seq_length,
            "id_file_utterance": id_file_utterance,
            "index": index,
        }


    def collate_fn(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        data: DataBatchPyChildes = DatasetPyChildes.collate_utterances(
            batch,
            token_id_pad=self.token_id_pad,
            max_len=self.max_len,
        )
        return data

    # def collate_fn_filter(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
    #     list_index_original = [item["index"] for item in batch]

    #     list_ok: list[bool] = [
    #         (self.dataframe.slice(i, 1).select(self.expr_keep).select(pl.all_horizontal(pl.all())).item())
    #         for i in list_index_original
    #     ]
    #     filtered_batch = [item for ok, item in zip(list_ok, batch) if ok]

    #     data: DataBatchPyChildes = collate(filtered_batch, token_id_pad=self.token_id_pad)
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

class TrainTokenMonitor(ptl.Callback):
    map_token_string: dict[int, str]
    counter_token: Counter[int]
    def __init__(self,
        *,
        tokens_to_watch: Sequence[str],
        tokenizer: hftk.Tokenizer,
    ):
        self.map_token_string = {}
        self.counter_token = Counter()
        for token in tokens_to_watch: 
            token_id: int | None = tokenizer.token_to_id(token)
            if token_id is None:
                logger.warning(f"Token '{token}' not found in tokenizer vocabulary and cannot be monitored.")
            self.map_token_string[token_id] = token
        
    def on_train_batch_start(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        batch: DataBatchPyChildes,
        batch_idx: int,
    ) -> None:
        input_ids = batch["input_ids"]
        dict_log = {}
        for token_id, token_str in self.map_token_string.items():
            count_token = (input_ids == token_id).sum().item()
            self.counter_token[token_id] += count_token
            dict_log[f"train/token_count/{token_str}"] = self.counter_token[token_id]

        pl_module.log_dict(dict_log,)


if __name__ == "__main__":
    from itertools import islice

    from torch.utils.data import DataLoader

    PATH_TRAIN = "./data/pychildes/pychildes-morphosyntax_postprocess.20260102_20251024.parquet"

    expr_keep = pl.col("features").list.contains(FEATURE_CAUSATIVE).not_()

    dataset = DatasetPyChildes(
        input_path=Path(PATH_TRAIN),
        token_id_pad=0,
        max_token_per_batch=1920,
        max_len=160,
        # expr_keep=expr_keep,
    )
    
    randgen_train_shuffling = torch.Generator()
    randgen_train_shuffling.manual_seed(23)
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.create_sampler(shuffle=randgen_train_shuffling),
        # batch_size=256,
        # shuffle=True,
        collate_fn=dataset.collate_fn,
        # generator=randgen_train_shuffling,
    )
    
    list_count_effective_token = []
    for i_batch, batch in enumerate(islice(dataloader, 10)):
        # input_ids = batch["input_ids"]
        # count_effective_token = ((input_ids != 0) & (input_ids != 1)).sum().item()
        # list_count_effective_token.append(count_effective_token)
        
        __import__("IPython").embed(using=False)
        # seq_length = batch["seq_length"]
        # list_count_effective_token.append(seq_length.float().sum())
    
    count_effective_token = torch.tensor(list_count_effective_token)
