from pathlib import Path
from typing import Any, TypedDict, cast

import lightning as ptl
import polars as pl
from torch.utils.data import Dataset as TorchDataset
from loguru import logger

from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data, strip_last_token
from verb_overgen.util.misc_util import ensure_dir


class DataOneScamp(DataOne, TypedDict):
    index_sentence: int
    """Index of the sentence. Each SCAMP entry (indexed by `index`) contains multiple sentences."""


class DataBatchScamp(DataBatch, TypedDict):
    index_sentence: list[int]


class DatasetScamp(TorchDataset[DataOneScamp]):
    def __init__(self, *, input_path: str, token_id_pad: int):
        self.dataframe = self._load_data(input_path)
        self.token_id_pad = token_id_pad

    def _load_data(self, input_path: str):
        dataset = pl.read_parquet(input_path).with_row_index("index_sentence")
        dataset = dataset.with_columns(
            pl.col("input_ids").pipe(strip_last_token).alias("input_ids"),
        )
        return dataset
    
    def debug_truncate(self, n_samples: int):
        print(f"Truncating SCAMP eval set to {n_samples} samples for debugging.")
        self.dataframe = self.dataframe.head(n_samples)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> DataOneScamp:
        input_ids = self.dataframe.item(index, "input_ids")
        data = cast(DataOneScamp, get_one_data(input_ids))
        data["index_sentence"] = self.dataframe.item(index, "index_sentence")
        return data

    def collate_fn(self, batch: list[DataOneScamp]) -> DataBatchScamp:
        data = cast(DataBatchScamp, collate(batch, token_id_pad=self.token_id_pad))
        data["index_sentence"] = [item["index_sentence"] for item in batch]
        return data


class EvalScampCallback(ptl.Callback):
    def __init__(self):
        self.eval_outputs: list[pl.DataFrame] = []
        self.eval_dir: Path

    def on_fit_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        self.eval_dir = ensure_dir(Path(trainer.default_root_dir) / "eval")

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
        output_path = self.eval_dir / f"eval_scamp_step_{epoch:02d}_{step:08d}.parquet"
        dataset_eval.write_parquet(output_path)
