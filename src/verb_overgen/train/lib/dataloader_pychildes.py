from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import polars as pl
from sympy import im
from torch.utils.data import Dataset
from loguru import logger

from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data


class DataBatchPyChildes(DataBatch, TypedDict):
    index: list[int]
    id_file_utterance: list[str]


class DataOnePyChildes(DataOne, TypedDict):
    index: int
    id_file_utterance: str


class DatasetPyChildes(Dataset[DataOnePyChildes]):
    dataframe: pl.DataFrame
    token_id_pad: int
    expr_keep: pl.Expr | Sequence[pl.Expr] | None

    def collate_fn(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        ...

    def __init__(
        self,
        *,
        input_path: Path,
        token_id_pad: int,
        expr_keep: pl.Expr | Sequence[pl.Expr] | None = None
    ):
        """
        Dataset for PyChildes data stored in a Parquet file.

        Args:
            input_path (Path): Path to the Parquet file containing the dataset.
            token_id_pad (int): Token ID used for padding sequences.
            expr_keep (pl.Expr | Sequence[pl.Expr] | None): Optional Polars expression(s) to filter data during collation. The expressions should evaluate to a boolean value, where True indicates the data should be included, and the expressions are combined using a logical AND operation. If None, no filtering is applied.
        """
        self.dataframe = self._load_data(input_path)
        self.token_id_pad = token_id_pad
        self.expr_keep = expr_keep
        if self.expr_keep is not None:
            self.collate_fn = self.collate_fn_filter
        else:
            self.collate_fn = self.collate_fn_no_filter

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

    def collate_fn_no_filter(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        data: DataBatchPyChildes = collate(batch, token_id_pad=self.token_id_pad) # type: ignore
        data["index"] = [item["index"] for item in batch]
        data["id_file_utterance"] = [item["id_file_utterance"] for item in batch]
        return data
    
    def collate_fn_filter(self, batch: list[DataOnePyChildes]) -> DataBatchPyChildes:
        list_index_original = [item["index"] for item in batch]

        list_ok: list[bool] = [
            (
                self.dataframe.slice(i, 1)
                .select(self.expr_keep)
                .select(pl.all_horizontal(pl.all()))
                .item()
            ) for i in list_index_original
        ]
        filtered_batch = [item for ok, item in zip(list_ok, batch) if ok] 

        data: DataBatchPyChildes = collate(filtered_batch, token_id_pad=self.token_id_pad) # type: ignore
        data["index"] = [item["index"] for item in filtered_batch]
        data["id_file_utterance"] = [item["id_file_utterance"] for item in filtered_batch]
        return data


if __name__ == "__main__":
    from itertools import islice
    from torch.utils.data import DataLoader

    PATH_TRAIN = "./data/pychildes/pychildes-morphosyntax_postprocess.20260102_20251024.parquet"

    expr_keep = pl.col("features").list.contains(FEATURE_CAUSATIVE).not_()

    dataset = DatasetPyChildes(
        input_path=Path(PATH_TRAIN),
        token_id_pad=0, 
        expr_keep=expr_keep,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    ) 

    for batch_idx, batch in enumerate(islice(dataloader, 10)):
        if (l := len(batch["index"])) < 512:
            print(f"batch {batch_idx}: {l}/512")
        pass
