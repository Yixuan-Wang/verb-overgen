from pathlib import Path
from typing import Any, TypedDict, cast

import lightning as ptl
import polars as pl
from loguru import logger
from torch.utils.data import Dataset as TorchDataset

from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data
from verb_overgen.util.misc_util import ensure_dir


class DataOneTestset(DataOne, TypedDict):
    index_sentence: int


class DataBatchTestset(DataBatch, TypedDict):
    index_sentence: list[int]


class DatasetTestset(TorchDataset[DataOneTestset]):
    def __init__(self, *, input_path: str, token_id_pad: int) -> None:
        self.dataframe = self._load_data(input_path)
        self.token_id_pad = token_id_pad

    def _load_data(self, input_path) -> pl.DataFrame:
        dataset = pl.read_parquet(input_path)
        assert "input_ids" in dataset.columns
        assert "index_sentence" in dataset.columns
        return dataset

    def debug_truncate(self, n_samples: int):
        logger.info(f"Truncating testset to {n_samples} samples for debugging.")
        self.dataframe = self.dataframe.head(n_samples)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> DataOneTestset:
        input_ids = self.dataframe.item(index, "input_ids")
        data = cast(DataOneTestset, get_one_data(input_ids))
        data["index_sentence"] = index
        return data

    def collate_fn(self, batch: list[DataOneTestset]) -> DataBatchTestset:
        data = collate(batch, token_id_pad=self.token_id_pad)
        data = cast(DataBatchTestset, data)
        data["index_sentence"] = [item["index_sentence"] for item in batch]
        return data


class TestTestsetOnceCallback(ptl.Callback):
    def __init__(self):
        self.test_outputs: list[pl.DataFrame] = []
        self.test_dir: Path
        self.test_dataset: pl.DataFrame

    def _ensure_dir(self, trainer: ptl.Trainer) -> None:
        self.test_dir = ensure_dir(Path(trainer.default_root_dir) / "test")

    def on_test_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        self._ensure_dir(trainer)
        self.test_outputs.clear()

    def on_test_batch_end(
        self,
        trainer: "ptl.Trainer",
        pl_module: "ptl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.test_outputs.append(outputs)

    def on_test_end(self, trainer: "ptl.Trainer", pl_module: "ptl.LightningModule") -> None:
        dataset = pl.concat(self.test_outputs)
        self.test_dataset = dataset


class TestTestsetCallback(TestTestsetOnceCallback):
    def on_test_end(self, trainer: "ptl.Trainer", pl_module: "ptl.LightningModule") -> None:
        step = trainer.global_step
        epoch = trainer.current_epoch
        dataset_eval = pl.concat(self.test_outputs).with_columns(
            pl.lit(step).alias("step"),
            pl.lit(epoch).alias("epoch"),
        )
        # Save to parquet with step information in filename
        output_path = self.test_dir / f"testset_step_{epoch:02d}_{step:08d}.parquet"
        dataset_eval.write_parquet(output_path)

class EvalTestsetCallback(ptl.Callback):
    def __init__(self):
        self.eval_outputs: list[pl.DataFrame] = []
        self.eval_dir: Path

    def on_fit_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        self.eval_dir = ensure_dir(Path(trainer.default_root_dir) / "test")

    def on_validation_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        self.eval_outputs.clear()

    def on_validation_batch_end(
        self,
        trainer: "ptl.Trainer",
        pl_module: "ptl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.eval_outputs.append(outputs)

    def on_validation_end(self, trainer: "ptl.Trainer", pl_module: "ptl.LightningModule") -> None:
        step = trainer.global_step
        epoch = trainer.current_epoch
        dataset_eval = pl.concat(self.eval_outputs).with_columns(
            pl.lit(step).alias("step"),
            pl.lit(epoch).alias("epoch"),
        )
        # Save to parquet with step information in filename
        output_path = self.eval_dir / f"testset_step_{epoch:02d}_{step:08d}.parquet"
        dataset_eval.write_parquet(output_path)