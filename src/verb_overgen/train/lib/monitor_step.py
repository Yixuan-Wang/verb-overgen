from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning as ptl
import polars as pl
import torch
from torch.nn.utils.rnn import unpad_sequence
from torch.utils.data import DataLoader as TorchDataLoader

from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.train.lib.dataloader_overgen import DataBatchOvergen, DatasetOvergen
from verb_overgen.train.lib.validate import last_sentence_index, nll
from verb_overgen.util.misc_util import ensure_dir

if TYPE_CHECKING:
    from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes


class MonitorStepEffect(ptl.Callback):
    """Measure per-batch training-step effect on eval loss for a target (verbs, feature) pair.

    Before and after each qualifying training batch, run eval on a subset dataloader
    filtered to `verbs`. Results are accumulated and written to <run_root>/eval/ at end of training.

    Throttling: a pre-eval only runs when at least `min_step_gap` steps have elapsed
    since the last post-eval (inclusive), preventing excessive overhead.
    """

    def __init__(
        self,
        *,
        verbs: list[str],
        feature: str = FEATURE_CAUSATIVE,
        dataset_train: "DatasetPyChildes",
        dataset_eval_overgen: DatasetOvergen,
        eval_batch_size: int,
        token_id_bos: int,
        min_step_gap: int = 25,
    ) -> None:
        self.verbs = verbs
        self.feature = feature
        self.dataset_dataframe = dataset_train.dataframe
        self.token_id_bos = token_id_bos
        self.min_step_gap = min_step_gap

        dataset_subset = dataset_eval_overgen.create_subset_verb(verbs)
        self.dataloader: TorchDataLoader[DataBatchOvergen] = torch.utils.data.DataLoader(  # type: ignore[assignment]
            dataset_subset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=dataset_subset.collate_fn,
        )

        self.last_post_step: int | None = None
        self.pending_post: bool = False
        self._pending_count: int = 0
        self.results: list[pl.DataFrame] = []

    def _count_feature_in_batch(self, batch_index: list[int]) -> int:
        return (
            self.dataset_dataframe[batch_index]
            .lazy()
            .select("featured_verbs")
            .explode("featured_verbs")
            .filter(pl.col("featured_verbs").is_not_null())
            .filter(
                (pl.col("featured_verbs").struct["feature"] == self.feature)
                & (pl.col("featured_verbs").struct["verb"].is_in(self.verbs))
            )
            .select(pl.len())
            .collect()
            .item()  # type: ignore[union-attr]
        )

    def on_train_batch_start(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        count = self._count_feature_in_batch(batch["index"])
        if count == 0:
            return

        if (
            self.last_post_step is not None
            and trainer.global_step - self.last_post_step < self.min_step_gap
        ):
            return

        df = self._run_eval(pl_module)
        df = df.with_columns(
            pl.lit(trainer.global_step).alias("step"),
            pl.lit(trainer.current_epoch).alias("epoch"),
            pl.lit("pre").alias("phase"),
            pl.lit(count).cast(pl.Int32).alias("feature_count"),
        )
        self.results.append(df)
        self.pending_post = True
        self._pending_count = count

    def on_train_batch_end(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.pending_post:
            return

        df = self._run_eval(pl_module)
        df = df.with_columns(
            pl.lit(trainer.global_step).alias("step"),
            pl.lit(trainer.current_epoch).alias("epoch"),
            pl.lit("post").alias("phase"),
            pl.lit(self._pending_count).cast(pl.Int32).alias("feature_count"),
        )
        self.results.append(df)
        self.last_post_step = trainer.global_step
        self.pending_post = False

    def _run_eval(self, pl_module: ptl.LightningModule) -> pl.DataFrame:
        pl_module.eval()
        outputs: list[pl.DataFrame] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch_on_device = {
                    k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                df = cast(pl.DataFrame, pl_module.validation_step(batch_on_device, batch_idx))
                list_input_ids = [
                    t.tolist()
                    for t in unpad_sequence(
                        batch["input_ids"].cpu(),
                        lengths=batch["seq_length"].cpu(),
                        batch_first=True,
                    )
                ]
                df = df.with_columns(
                    pl.Series("input_ids", list_input_ids, dtype=pl.List(pl.Int64))
                )
                outputs.append(df)
        pl_module.train()

        return pl.concat(outputs).with_columns(
            pl.col("token_prob")
            .list.slice(last_sentence_index(pl.col("input_ids"), self.token_id_bos))
            .pipe(nll)
            .alias("nll_last_sentence"),
        ).drop("token_prob", "input_ids")

    def on_fit_end(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        if not self.results:
            return
        df = pl.concat(self.results)
        out_dir = ensure_dir(Path(trainer.default_root_dir) / "eval")
        df.write_parquet(out_dir / f"monitor_step_{self.feature}.parquet")
