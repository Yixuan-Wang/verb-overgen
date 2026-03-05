from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import lightning as ptl
import polars as pl
import torch
import wandb
from apfel import not_none
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from verb_overgen.train.lib.dataloader import DataBatch, DataOne, collate, get_one_data, strip_last_token
from verb_overgen.train.lib.validate import ci, nll, VERB_TARGET_5
from verb_overgen.util.misc_util import ensure_dir


class DataOneOvergen(DataOne, TypedDict):
    index_sentence: int
    verb: str
    kind: str


class DataBatchOvergen(DataBatch, TypedDict):
    index_sentence: list[int]
    verb: list[str]
    kind: list[str]

def clean_dataset_overgen(dataset: pl.DataFrame) -> pl.DataFrame:
    assert "input_ids" in dataset.columns
    assert "index_sentence" in dataset.columns
    assert "verb" in dataset.columns
    assert "kind" in dataset.columns

    dataset = dataset.filter(pl.col("kind") != "sentence_context")

    dataset = dataset.with_columns(
        pl.col("input_ids").pipe(strip_last_token).alias("input_ids"),
    )
    return dataset

class DatasetOvergen(TorchDataset[DataOneOvergen]):
    def __init__(self, *, dataframe: pl.DataFrame, token_id_pad: int) -> None:
        self.dataframe = dataframe
        self.token_id_pad = token_id_pad

    @classmethod
    def from_path(cls, input_path: str, token_id_pad: int) -> "DatasetOvergen":
        dataframe = cls._load_data(input_path)
        return cls(dataframe=dataframe, token_id_pad=token_id_pad)

    def create_subset_verb(self, verb_list: None | Sequence[str] = None) -> "DatasetOvergen":
        """Create a subset of the dataset for a small subset of verbs."""
        if verb_list is None:
            verb_list = VERB_TARGET_5

        dataframe_subset = self.dataframe.filter(
            pl.col("verb").is_in(verb_list),
        )
        return DatasetOvergen(dataframe=dataframe_subset, token_id_pad=self.token_id_pad)

    @classmethod
    def _load_data(cls, input_path) -> pl.DataFrame:
        dataset = pl.read_parquet(input_path)
        dataset = clean_dataset_overgen(dataset)
        return dataset

    def create_subset_for_debug(self, n_samples: int):
        logger.info(f"Truncating overgen dataset to {n_samples} samples for debugging.")
        dataframe = self.dataframe.head(n_samples)
        return DatasetOvergen(dataframe=dataframe, token_id_pad=self.token_id_pad)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> DataOneOvergen:
        input_ids = self.dataframe.item(index, "input_ids")
        data = cast(DataOneOvergen, get_one_data(input_ids))
        data["index_sentence"] = self.dataframe.item(index, "index_sentence") 
        data["verb"] = self.dataframe.item(index, "verb")
        data["kind"] = self.dataframe.item(index, "kind")
        return data

    def collate_fn(self, batch: list[DataOneOvergen]) -> DataBatchOvergen:
        data = collate(batch, token_id_pad=self.token_id_pad)
        data = cast(DataBatchOvergen, data)
        data["index_sentence"] = [item["index_sentence"] for item in batch]
        data["verb"] = [item["verb"] for item in batch]
        data["kind"] = [item["kind"] for item in batch]
        return data


class EvalOvergenMonitor(ptl.Callback):
    def __init__(self, *, dataloader: TorchDataLoader[DataBatchOvergen], run_every_n_steps: int = 100):
        self.dataloader = dataloader
        self.run_every_n_steps = run_every_n_steps

    def on_train_batch_end(
        self,
        trainer: "ptl.Trainer",
        pl_module: "ptl.LightningModule",
        *_,
        **__,
    ) -> None:
        if (trainer.global_step + 1) % self.run_every_n_steps != 0:
            return

        pl_module.eval()
        outputs: list[pl.DataFrame] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch_on_device = {
                    k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

                output = cast(pl.DataFrame, pl_module.validation_step(batch_on_device, batch_idx))
                outputs.append(output)
        pl_module.train()

        if not outputs:
            raise ValueError("No outputs collected from validation_step.")

        df_results: pl.DataFrame = pl.concat(outputs)
        df_results = (
            df_results.with_columns(
                pl.col("token_prob").pipe(nll).alias("nll"),
                pl.col("kind").str.strip_prefix("sentence_"),
            )
            .group_by("verb", "kind")
            .agg(
                pl.col("nll").mean().alias("nll_mean"),
                pl.col("nll").alias("nll"),
            )
            .sort("verb", "kind")
        )

        dict_results = {}
        for verb, kind, nll_mean, in zip(
            df_results["verb"],
            df_results["kind"],
            df_results["nll_mean"],
            # df_results["nll"],
        ):
            # ci_lower, ci_upper = ci(nll_series.to_numpy())
            dict_results[f"eval/loss_mean/{verb}_{kind}"] = nll_mean
            # dict_results[f"eval/loss_ci_lower/{verb}_{kind}"] = ci_lower
            # dict_results[f"eval/loss_ci_upper/{verb}_{kind}"] = ci_upper

        pl_module.log_dict(dict_results)


class TestOvergenOnceCallback(ptl.Callback):
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


class TestOvergenCallback(TestOvergenOnceCallback):
    def on_test_end(self, trainer: "ptl.Trainer", pl_module: "ptl.LightningModule") -> None:
        step = trainer.global_step
        epoch = trainer.current_epoch
        dataset_eval = pl.concat(self.test_outputs).with_columns(
            pl.lit(step).alias("step"),
            pl.lit(epoch).alias("epoch"),
        )
        # Save to parquet with step information in filename
        output_path = self.test_dir / f"overgen_step_{epoch:02d}_{step:08d}.parquet"
        dataset_eval.write_parquet(output_path)


class EvalOvergenCallback(ptl.Callback):
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
        output_path = self.eval_dir / f"overgen_step_{epoch:02d}_{step:08d}.parquet"
        dataset_eval.write_parquet(output_path)
