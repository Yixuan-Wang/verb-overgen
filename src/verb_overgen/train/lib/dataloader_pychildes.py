from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Generator, Sequence, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, Required, TypedDict, cast

import lightning.pytorch as ptl
import polars as pl
import tokenizers as hftk
import torch
from jaxtyping import Int
from loguru import logger
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED, FILE_ID_AUGMENTED
from verb_overgen.train.lib.dataloader import DataBatch, DataOne, get_one_data, strip_last_token
from verb_overgen.util.pl_util import sample as _lazy_sample


class AugmentationSource(TypedDict, total=False):
    """A single augmentation parquet pair to append to the main training data.

    The parquets must share the schemas of the main/annotation training files
    (see `prep.augmentation_preprocess`). `id_file` values must not collide with
    those in the main dataset — the `{stem}/{line_number}` convention from
    `augmentation_preprocess` guarantees this for distinct stems.
    """

    path_main: Required[Path]
    path_annotation: Required[Path]
    size: int | None
    """Optional cap on the number of rows sampled from this source. `None` means use all rows."""


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

    n_main: int = 0
    """Number of non-augmentation rows. Augmentation rows are appended as a
    contiguous suffix, so the main partition is `dataframe[:n_main]` and the
    aug partition is `dataframe[n_main:]`."""

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
        augmentation_sources: Sequence[AugmentationSource] = (),
        augmentation_seed: int | None = None,
    ) -> DatasetPyChildes:
        df, n_main = cls._load_data(
            input_path_main=input_path_main,
            input_path_annotation=input_path_annotation,
            max_len=max_len,
            augmentation_sources=augmentation_sources,
            augmentation_seed=augmentation_seed,
        )
        return cls(
            df,
            token_id_pad=token_id_pad,
            max_len=max_len,
            max_token_per_batch=max_token_per_batch,
            n_main=n_main,
        )

    @classmethod
    def _load_data(
        cls,
        input_path_main: Path,
        input_path_annotation: Path,
        *,
        max_len: int,
        augmentation_sources: Sequence[AugmentationSource] = (),
        augmentation_seed: int | None = None,
    ):
        lf_main = pl.scan_parquet(input_path_main)
        lf_annotation = pl.scan_parquet(input_path_annotation)

        n_main = int(lf_main.select(pl.len()).collect().item()) # type: ignore

        if augmentation_sources:
            lf_aug_main_parts: list[pl.LazyFrame] = []
            lf_aug_annotation_parts: list[pl.LazyFrame] = []
            for src in augmentation_sources:
                lf_src_main = pl.scan_parquet(src["path_main"])
                lf_src_annotation = pl.scan_parquet(src["path_annotation"])

                size = src.get("size")
                if size is not None:
                    lf_src_main = _lazy_sample(lf_src_main, size=size, seed=augmentation_seed)
                    # The left join drops annotation rows whose `id_file` was not
                    # sampled, so no separate sampling is needed on the annotation side.

                logger.info(
                    f"Augmentation source {src['path_main'].name}: "  # noqa: ISC002
                    f"{'sampled ' + str(size) if size is not None else 'all rows'}"  # noqa: ISC002
                    f"{' (seed=' + str(augmentation_seed) + ')' if size is not None and augmentation_seed is not None else ''}"  # noqa: ISC002
                )
                lf_aug_main_parts.append(lf_src_main)
                lf_aug_annotation_parts.append(lf_src_annotation)

            lf_main = pl.concat([lf_main, *lf_aug_main_parts], how="vertical_relaxed")
            lf_annotation = pl.concat(
                [lf_annotation, *lf_aug_annotation_parts], how="vertical_relaxed"
            )

        logger.info("Joining main dataset and annotation dataset.")
        df = (
            lf_main
            .join(lf_annotation, on=["id_file", "id_utterance"], how="left", coalesce=True, maintain_order="left")
            .with_columns(
                pl.col("features").fill_null([]),
                pl.col("featured_verbs").fill_null([]),
            )
            .collect()
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

        return df, n_main

    def debug_truncate(self, n_samples: int):
        logger.info(f"Truncating training set to {n_samples} samples for debugging.")
        self.dataframe = self.dataframe.head(n_samples)
        self.n_main = min(self.n_main, len(self.dataframe))

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

        Main (non-aug) rows are greedily binned against `max_token_per_batch`.
        Augmentation rows (whose `id_file` starts with `FILE_ID_AUGMENTED`) are
        dispersed at random positions into the bins *after* the bin-level
        shuffle, using the same generator as the shuffle. The bin token budget
        is not respected for these insertions — aug samples are expected to be
        short.

        Args:
            shuffle (bool | torch.Generator | None): If True, shuffle the bins. If a torch.Generator is provided, use it for shuffling. If False or None, do not shuffle.
        """
        main_len = (
            self.dataframe.lazy()
            .head(self.n_main)
            .select(pl.col("input_ids").list.len())
            .collect()
            .to_series()
            .to_list()
        )
        n_aug = len(self.dataframe) - self.n_main

        return SamplerBin(
            main_len,
            n_aug=n_aug,
            shuffle=shuffle,
            max_token_per_batch=self.max_token_per_batch,
        )

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
        # Track the last *non-aug* file in the current pack. Aug rows don't
        # update this, so a real non-aug→non-aug file boundary still forces a
        # break even when one or more aug rows sit between the two files.
        _prev_non_aug_file: str | None = None
        list_ranges: list[tuple[int, int]] = []
        for _idx, (_size, _id_file) in enumerate(zip(list_seq_length_flat, list_id_file_flat, strict=True)):
            if _size > max_len:
                raise ValueError(f"Sequence at index {_idx} exceeds max_len {max_len} with size {_size}.")

            _is_aug = _id_file.startswith(FILE_ID_AUGMENTED)
            _file_boundary = (
                not _is_aug
                and _prev_non_aug_file is not None
                and _id_file != _prev_non_aug_file
            )
            if _acc + _size > max_len or _file_boundary:
                list_ranges.append((_start, _idx))
                _acc, _start = 0, _idx
                _prev_non_aug_file = None

            if not _is_aug:
                _prev_non_aug_file = _id_file
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
    _ranges: list[tuple[int, int]]
    _n_main: int
    _n_aug: int
    _shuffle: bool
    _generator: torch.Generator | None

    def __init__(
        self,
        list_len: Sequence[int],
        *,
        max_token_per_batch: int,
        shuffle: bool | torch.Generator | None = None,
        n_aug: int = 0,
    ):
        """Bin samples greedily and optionally shuffle and/or disperse aug rows.

        The main rows occupy dataframe indices `[0, len(list_len))`; aug rows
        are assumed to be a contiguous suffix at indices `[len(list_len),
        len(list_len) + n_aug)`. Bins are stored as `(start, end)` ranges in
        `__init__`; the bin permutation and aug-row dispersion are recomputed
        on every `__iter__` call so that each epoch sees a fresh order. The
        cached `torch.Generator` is reused across epochs, so its RNG state
        advances naturally.

        Args:
            list_len: Lengths of the *main* rows to be greedily binned.
            max_token_per_batch: Greedy cap on cumulative length per bin.
            shuffle: If truthy, permute the bin order on each `__iter__`. If
                a `torch.Generator` is provided, it is reused as the source
                of randomness for both shuffling and aug dispersion. If
                falsy but `n_aug > 0`, a fresh `torch.Generator` is created
                for dispersion only.
            n_aug: Number of augmentation rows to disperse into the bins at
                random positions *after* the bin shuffle. Bin token budgets
                are not respected for these insertions.
        """
        self._ranges = list(
            self._create_bin_length(list_len, max_token_per_batch=max_token_per_batch)
        )
        self._n_main = len(list_len)
        self._n_aug = n_aug
        self._shuffle = bool(shuffle)

        if shuffle:
            self._generator = torch.Generator() if shuffle is True else shuffle
        else:
            if n_aug > 0:
                raise ValueError("Augmentation rows cannot be dispersed without shuffling. Set `shuffle` to True or provide a torch.Generator.")
            self._generator = None

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

    def _disperse_aug_inserts(self, order: list[int]) -> list[list[tuple[int, int]]]:
        """Pick random insertion positions for aug rows across the post-shuffle bins.

        Returns a list parallel to `order`: entry `i` holds the
        `(position, df_idx)` insertions for the `i`-th post-shuffle bin.
        """
        n_bins = len(order)
        aug_inserts: list[list[tuple[int, int]]] = [[] for _ in range(n_bins)]
        if self._n_aug == 0 or n_bins == 0:
            return aug_inserts

        gen = self._generator
        assert gen is not None, "A torch.Generator must be provided for augmentation row dispersion."
        bin_choice = torch.randint(0, n_bins, (self._n_aug,), generator=gen).tolist()
        running_sizes = [self._ranges[order[i]][1] - self._ranges[order[i]][0] for i in range(n_bins)]
        for aug_offset, shuf_bin in enumerate(bin_choice):
            pos = int(torch.randint(0, running_sizes[shuf_bin] + 1, (1,), generator=gen).item())
            aug_inserts[shuf_bin].append((pos, self._n_main + aug_offset))
            running_sizes[shuf_bin] += 1
        return aug_inserts

    def __iter__(self):
        n_bins = len(self._ranges)

        if self._shuffle:
            assert self._generator is not None, "A torch.Generator must have been provided for shuffling."
            order = torch.randperm(n_bins, generator=self._generator).tolist()
        else:
            order = list(range(n_bins))

        aug_inserts = self._disperse_aug_inserts(order)

        for shuf_idx, orig_idx in enumerate(order):
            s, e = self._ranges[orig_idx]
            bin_list = list(range(s, e))
            for pos, aug_df_idx in aug_inserts[shuf_idx]:
                bin_list.insert(pos, aug_df_idx)
            yield bin_list

    def __len__(self):
        return len(self._ranges)


class TrainTokenMonitor(ptl.Callback):
    map_token_string: dict[int, str]
    counter_token: Counter[int]
    map_index_verb_features: dict[int, list[tuple[str, str]]]
    counter_verb_feature: Counter[tuple[str, str]]
    primary_verb: str | None

    def __init__(
        self,
        *,
        tokens_to_watch: Sequence[str],
        tokenizer: hftk.Tokenizer,
        dataset: DatasetPyChildes,
    ):
        self.map_token_string = {}
        self.counter_token = Counter()
        for token in tokens_to_watch:
            token_id: int | None = tokenizer.token_to_id(token)
            if token_id is None:
                logger.warning(f"Token '{token}' not found in tokenizer vocabulary and cannot be monitored.")
            self.map_token_string[token_id] = token

        self.primary_verb = tokens_to_watch[0] if tokens_to_watch else None

        # Pre-compute mapping from dataset index to relevant (verb, feature) pairs
        self.counter_verb_feature = Counter()
        self.map_index_verb_features = defaultdict(list)

        dataframe_watched_features = cast(
            pl.DataFrame,
            dataset.dataframe.lazy().select(
                "index",
                pl.col("featured_verbs").list.filter(pl.element().struct["verb"].is_in(tokens_to_watch)),
            ).filter(pl.col("featured_verbs").list.len() > 0).explode("featured_verbs").collect()
        )

        for _index, _featured_verb in zip(dataframe_watched_features["index"], dataframe_watched_features["featured_verbs"]):
            _verb = _featured_verb["verb"]
            _feature = _featured_verb["feature"]
            self.map_index_verb_features[_index].append((_verb, _feature))

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
            if token_str == self.primary_verb:
                dict_log["train/token_count/this"] = self.counter_token[token_id]

        for idx in batch["index"]:
            for verb, feature in self.map_index_verb_features.get(idx, ()):
                self.counter_verb_feature[(verb, feature)] += 1

        for (verb, feature), count in self.counter_verb_feature.items():
            dict_log[f"train/verb_feature/{verb}/{feature}"] = count
            if verb == self.primary_verb:
                dict_log[f"train/verb_feature/this/{feature}"] = count

        pl_module.log_dict(
            dict_log,
        )

def remove_preemption(dataset: DatasetPyChildes):
    df_filter = dataset.dataframe.filter(
        pl.col("features").list.contains(FEATURE_CAUSATIVE)
    )
    dataset.index_should_remove.update(df_filter["index"])
    
def remove_preemption_on_verb(dataset: DatasetPyChildes, *, verb: str | Iterable[str]):
    if isinstance(verb, str):
        expr_filter_verb = pl.element().struct["verb"].eq(verb)
    else:
        expr_filter_verb = pl.element().struct["verb"].is_in(list(verb))
    
    df_filter = dataset.dataframe.filter(
        pl.col("featured_verbs").list.eval(
            pl.element().struct["feature"].eq(FEATURE_CAUSATIVE)
            & expr_filter_verb
        ).list.any()
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

def remove_entrenchment_on_verb(dataset: DatasetPyChildes, *, verb: str | Iterable[str], randgen: torch.Generator | None):
    """
    Warning:
        This actually removes non-preemption entrenchment samples.

    Usage:
        ```python
        remove_entrenchment_on_verb(dataset, verb="laugh", randgen=torch.Generator().manual_seed(42))
        ```
    """
    if isinstance(verb, str):
        expr_filter_verb = pl.element().struct["verb"].eq(verb)
    else:
        expr_filter_verb = pl.element().struct["verb"].is_in(list(verb))
    
    df_causative = dataset.dataframe.filter(
        pl.col("featured_verbs").list.eval(
            pl.element().struct["feature"].eq(FEATURE_CAUSATIVE)
            & expr_filter_verb
        ).list.any()
    )
    
    df_causative_count = (
        df_causative.select(
            pl.col("featured_verbs").list.filter(
                pl.element().struct["feature"].eq(FEATURE_CAUSATIVE)
                & expr_filter_verb
            )
        )
        .explode("featured_verbs")
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


    # with __import__("ipdb").launch_ipdb_on_exception():
    #     remove_token(dataset, token_make)

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
    from verb_overgen.train.lib.validate import VERB_TARGET_5
    
    df_verb_base = pl.DataFrame({"verb": VERB_TARGET_5})
    list_df_verb_count = []

    for i_batch, batch in enumerate(dataloader):
        # input_ids = batch["input_ids"]
        # count_effective_token = ((input_ids != 0) & (input_ids != 1)).sum().item()
        # list_count_effective_token.append(count_effective_token)
        # if batch.get("count_removed", 0) > 0:
        #     __import__("IPython").embed(using=False)

        # count_removed += batch.get("count_removed", 0)
        batch_index: list[int] = batch["index"]
        df_verb_count = dataset.dataframe[batch_index].select("featured_verbs").explode("featured_verbs").filter(
            pl.col("featured_verbs").is_null().not_()
        ).select(pl.col("featured_verbs").struct["verb"]).filter(pl.col("verb").is_in(VERB_TARGET_5)).select(
            pl.col("verb").value_counts().struct.unnest()
        )
        
        list_df_verb_count.append(df_verb_count.join(df_verb_base, on="verb", how="right").fill_null(0).with_columns(
            pl.lit(i_batch).alias("batch_index")
        ))
        
    df_verb_count = pl.concat(list_df_verb_count)
    
    __import__("IPython").embed(using=False)
        # dataset.dataframe.[]

    print(count_removed)
