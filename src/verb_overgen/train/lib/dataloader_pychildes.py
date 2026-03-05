from __future__ import annotations

from collections import Counter
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import lightning.pytorch as ptl
import polars as pl
import tokenizers as hftk
import torch
from jaxtyping import Int
from loguru import logger
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.train.lib.dataloader import DataBatch, DataOne, get_one_data, strip_last_token


class DataBatchPyChildes(DataBatch, TypedDict):
    index: list[int]
    id_file_utterance: list[str]
    attention_mask: Int[Tensor, "batch seq"]
    count_removed: NotRequired[int]


class DataOnePyChildes(DataOne, TypedDict):
    index: int
    id_file: list[str]
    id_file_utterance: str


@dataclass
class DatasetPyChildes(Dataset[DataOnePyChildes]):
    """A training set curated from PyChildes.

    Warning:
        Always call `from_path` to init the dataset to ensure the preprocessing steps are applied.
    """

    dataframe: pl.DataFrame
    """The underlying Polars DataFrame containing the dataset. It should have the following columns:
        - `input_ids`: A list of token IDs representing the tokenized input sequence.
        - `id_file`: The identifier for the source file of the utterance.
        - `id_utterance`: The identifier for the specific utterance within the file.
    """

    token_id_pad: int
    """Token ID used for padding sequences."""

    max_len: int
    """Maximum sequence length allowed."""

    max_token_per_batch: int
    """Maximum number of tokens allowed per batch during sampling."""

    index_should_remove: set[int] = field(default_factory=set)
    """A set of indices corresponding to samples that should be removed during training."""

    @classmethod
    def from_path(
        cls,
        *,
        input_path_main: Path,
        input_path_annotation: Path,
        token_id_pad: int,
        max_len: int,
        max_token_per_batch: int,
    ) -> DatasetPyChildes:
        return cls(
            cls._load_data(input_path_main=input_path_main, input_path_annotation=input_path_annotation, max_len=max_len),
            token_id_pad=token_id_pad,
            max_len=max_len,
            max_token_per_batch=max_token_per_batch,
        )

    @classmethod
    def _load_data(cls, input_path_main: Path, input_path_annotation: Path, *, max_len: int):
        df_main = pl.read_parquet(input_path_main)
        df_annotation = pl.read_parquet(input_path_annotation)

        num_rows = len(df_main)

        logger.info("Joining main dataset and annotation dataset.")
        # df = df_main.join(df_annotation, on=["id_file", "id_utterance"], validate="1:1", coalesce=True)
        assert num_rows == len(df_annotation), "The main dataset and annotation dataset must have the same number of rows."
        assert df_main[0, "id_file"] == df_annotation[0, "id_file"] and df_main[0, "id_utterance"] == df_annotation[0, "id_utterance"], "The main dataset and annotation dataset must be aligned in the same order."
        assert df_main[num_rows - 1, "id_file"] == df_annotation[num_rows - 1, "id_file"] and df_main[num_rows - 1, "id_utterance"] == df_annotation[num_rows - 1, "id_utterance"], "The main dataset and annotation dataset must be aligned in the same order."
        df = df_main.with_columns(
            df_annotation["features"],
            df_annotation["featured_verbs"],
        )

        logger.info("Processing dataset for training.")
        if df.schema.get("input_ids") != pl.List(pl.Int32):
            raise KeyError(
                "Pre-tokenize dataset first using `train.run_tokenize`. The `input_ids` column should be of type `List(Int32)`."
            )
        if "features" not in df.columns or "featured_verbs" not in df.columns:
            raise KeyError("The dataset must contain 'features' and 'featured_verbs' columns for filtering.")

        if "index" not in df.columns:
            df = df.select(
                pl.concat_str(pl.col("id_file"), pl.lit(":"), pl.col("id_utterance")).alias("id_file_utterance"),
                pl.all(),
            )

        # Note: The filter and preprocess function are moved to `train.run_tokenize.load_data`.
        df = df.with_columns(pl.col("input_ids").pipe(strip_last_token).cast(pl.List(pl.Int64))).with_row_index("index")

        if df.select(pl.col("input_ids").list.len().max()).item() > max_len:
            raise ValueError(f"Maximum sequence length in dataset exceeds max_len={max_len}")


        return df

    def debug_truncate(self, n_samples: int):
        logger.info(f"Truncating training set to {n_samples} samples for debugging.")
        self.dataframe = self.dataframe.head(n_samples)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> DataOnePyChildes:
        input_ids = self.dataframe.item(index, "input_ids")
        data: DataOnePyChildes = get_one_data(input_ids)  # type: ignore
        data["index"] = self.dataframe.item(index, "index") # this is the id
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
            if _acc + _size > max_len or _id_file != _prev_file:
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
            [
                torch.cat([batch[_idx_flat]["input_ids"] for _idx_flat in range(_start, _end)], dim=0)
                for _start, _end in list_ranges
            ],
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

    def collate_fn_without_filter(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        data: DataBatchPyChildes = DatasetPyChildes.collate_utterances(
            batch,
            token_id_pad=self.token_id_pad,
            max_len=self.max_len,
        )
        return data

    def collate_fn_with_filter(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        list_index_original = [item["index"] for item in batch]

        filtered_batch = [item for index, item in zip(list_index_original, batch) if index not in self.index_should_remove]
        count_removed = len(batch) - len(filtered_batch)

        data: DataBatchPyChildes = DatasetPyChildes.collate_utterances(
            filtered_batch,
            token_id_pad=self.token_id_pad,
            max_len=self.max_len,
        )
        data["count_removed"] = count_removed
        return data


class SamplerBin(Sampler[list[int]]):
    bin_length: list[tuple[int, int]]

    def __init__(
        self, list_len: Sequence[int], *, max_token_per_batch: int, shuffle: bool | torch.Generator | None = None
    ):
        self.bin_length = list(self._create_bin_length(list_len, max_token_per_batch=max_token_per_batch))

        if shuffle:
            generator = torch.Generator() if shuffle is True else shuffle
            indices = torch.randperm(len(self.bin_length), generator=generator).tolist()
            self.bin_length = [self.bin_length[i] for i in indices]

    @staticmethod
    def _create_bin_length(
        list_len: Sequence[int], *, max_token_per_batch: int
    ) -> Generator[tuple[int, int], None, None]:
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

    def __init__(
        self,
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

        pl_module.log_dict(
            dict_log,
        )

def remove_preemption(dataset: DatasetPyChildes):
    df_filter = dataset.dataframe.filter(
        pl.col("features").list.contains(FEATURE_CAUSATIVE)
    )
    dataset.index_should_remove.update(df_filter["index"])

def remove_entrenchment(dataset: DatasetPyChildes, *, randgen: torch.Generator | None):
    """
    Warning:
        This actually removes all non-preemption entrenchment samples.

    Usage:
        ```python
        remove_entrenchment(dataset, randgen=torch.Generator().manual_seed(42))
        ```
    """
    df_causative_count = (
        dataset.dataframe.filter(pl.col("features").list.contains(FEATURE_CAUSATIVE))
        .select("featured_verbs")
        .explode("featured_verbs")
        .filter(pl.col("featured_verbs").struct["feature"] == FEATURE_CAUSATIVE)
        .select(
            pl.col("featured_verbs").struct["verb"].value_counts().struct.unnest()
        )
    )

    for verb, count in zip(df_causative_count["verb"], df_causative_count["count"]):
        seed = torch.randint(0, 1_000_000, (1,)).item() if randgen else None
        df_entrenched = dataset.dataframe.filter(
            pl.col("featured_verbs").list.eval(
                pl.element().struct["feature"].eq(FEATURE_CAUSATIVE_ENTRENCHED) & pl.element().struct["verb"].eq(verb)
            ).list.any(),
        )
        if len(df_entrenched) < count:
            logger.warning(f"Not enough entrenched (but not preemptive) samples for verb '{verb}' to match the count of preemptive samples. Requested {count}, but only {len(df_entrenched)} available. ")
            continue

        df_sample = df_entrenched.sample(count, seed=seed)
        dataset.index_should_remove.update(df_sample["index"])

def remove_token(dataset: DatasetPyChildes, token_id: int, *, sample_frac: float = 1.0, seed: int | None = None):
    """
    Remove samples containing a specific token.

    Args:
        dataset: The dataset from which to remove samples.
        token_id: The token ID to filter by. All samples containing this token will be removed.
        sample_frac: If less than 1.0, randomly sample a fraction of the matching samples to remove instead of all. Must be between 0 and 1.
        seed: An optional random seed for reproducibility when sampling. Ignored if `sample_frac` is 1.0.
    """
    df_filter = dataset.dataframe.filter(pl.col("input_ids").list.contains(token_id))

    if sample_frac < 1.0:
        df_filter = df_filter.sample(fraction=sample_frac, seed=seed)

    dataset.index_should_remove.update(df_filter["index"])

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    PATH_MAIN = "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"
    PATH_ANNOTATION = "./data/pychildes/pychildes-morphosyntax_annotation-20260225.20251024.parquet"

    dataset = DatasetPyChildes.from_path(
        input_path_main=Path(PATH_MAIN),
        input_path_annotation=Path(PATH_ANNOTATION),
        token_id_pad=0,
        max_token_per_batch=8192,
        max_len=512,
    )

    from verb_overgen.train.lib.utils import load_train_config
    from verb_overgen.train.train import TrainingSession

    _session = TrainingSession(load_train_config())
    tokenizer = _session.setup_tokenizer()

    token_make: int = tokenizer._tokenizer.token_to_id("make")


    with __import__("ipdb").launch_ipdb_on_exception():
        remove_token(dataset, token_make)

    randgen_train_shuffling = torch.Generator()
    randgen_train_shuffling.manual_seed(23)
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.create_sampler(shuffle=randgen_train_shuffling),
        # batch_size=256,
        # shuffle=True,
        collate_fn=dataset.collate_fn_with_filter,
        # generator=randgen_train_shuffling,
    )

    count_removed = 0
    for i_batch, batch in enumerate(dataloader):
        # input_ids = batch["input_ids"]
        # count_effective_token = ((input_ids != 0) & (input_ids != 1)).sum().item()
        # list_count_effective_token.append(count_effective_token)
        if batch.get("count_removed", 0) > 0:
            __import__("IPython").embed(using=False)

        count_removed += batch.get("count_removed", 0)
        # seq_length = batch["seq_length"]
        # list_count_effective_token.append(seq_length.float().sum())

    print(count_removed)
