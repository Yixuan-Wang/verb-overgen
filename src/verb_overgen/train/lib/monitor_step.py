from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning as ptl
import polars as pl
import torch
from torch.nn.utils.rnn import unpad_sequence
from torch.utils.data import DataLoader as TorchDataLoader

from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.train.lib.dataloader_overgen import DataBatchOvergen, DatasetOvergen
from verb_overgen.train.lib.validate import last_sentence_index, nll
from verb_overgen.util.misc_util import ensure_dir

if TYPE_CHECKING:
    from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes


@dataclass
class _PendingVerb:
    verb: str
    step_type: str
    trigger_reason: str
    feature_count_causative: int
    feature_count_noncausative: int


class MonitorStepEffect(ptl.Callback):
    """Measure per-batch training-step effect on eval loss, separately per verb.

    For each verb, tracks two kinds of steps:
    - causative: batch contains FEATURE_CAUSATIVE occurrences of the verb (always triggers)
    - non_causative: batch contains FEATURE_CAUSATIVE_ENTRENCHED occurrences.
      Triggered by matching (noncausative_evals < causative_evals) or reservoir
      (every reservoir_stride qualifying non-causative batches).

    Results are written to <run_root>/eval/monitor_step_<verb>.parquet at end of training.
    """

    def __init__(
        self,
        *,
        verbs: list[str],
        dataset_train: "DatasetPyChildes",
        dataset_eval_overgen: DatasetOvergen,
        eval_batch_size: int,
        token_id_bos: int,
        reservoir_stride: int = 50,
    ) -> None:
        self.verbs = verbs
        self.dataset_dataframe = dataset_train.dataframe
        self.token_id_bos = token_id_bos
        self.reservoir_stride = reservoir_stride

        self.dataloaders: dict[str, TorchDataLoader] = {}
        for verb in verbs:
            subset = dataset_eval_overgen.create_subset_verb([verb])
            self.dataloaders[verb] = torch.utils.data.DataLoader(
                subset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=subset.collate_fn,
            )

        self.causative_evals_run: dict[str, int] = {v: 0 for v in verbs}
        self.noncausative_evals_run: dict[str, int] = {v: 0 for v in verbs}
        self.batches_since_reservoir: dict[str, int] = {v: 0 for v in verbs}

        self._pending: list[_PendingVerb] = []
        self.results: dict[str, list[pl.DataFrame]] = {v: [] for v in verbs}

    def _count_features_per_verb(self, batch_index: list[int]) -> dict[str, tuple[int, int]]:
        """Return {verb: (causative_count, noncausative_count)} for all watched verbs."""
        raw = (
            self.dataset_dataframe[batch_index]
            .lazy()
            .select("featured_verbs")
            .explode("featured_verbs")
            .filter(pl.col("featured_verbs").is_not_null())
            .filter(pl.col("featured_verbs").struct["verb"].is_in(self.verbs))
            .select(
                pl.col("featured_verbs").struct["verb"].alias("verb"),
                pl.col("featured_verbs").struct["feature"].alias("feature"),
            )
            .group_by("verb", "feature")
            .agg(pl.len().alias("cnt"))
            .collect()
        )

        counts: dict[str, tuple[int, int]] = {v: (0, 0) for v in self.verbs}
        for row in raw.iter_rows(named=True):
            v, feat, cnt = row["verb"], row["feature"], row["cnt"]
            c, n = counts[v]
            if feat == FEATURE_CAUSATIVE:
                counts[v] = (c + cnt, n)
            elif feat == FEATURE_CAUSATIVE_ENTRENCHED:
                counts[v] = (c, n + cnt)
        return counts

    def _decide_trigger(self, verb: str, c_v: int, n_v: int) -> _PendingVerb | None:
        if c_v > 0:
            return _PendingVerb(
                verb=verb,
                step_type="causative",
                trigger_reason="causative",
                feature_count_causative=c_v,
                feature_count_noncausative=n_v,
            )
        if n_v > 0:
            self.batches_since_reservoir[verb] += 1
            if self.noncausative_evals_run[verb] < self.causative_evals_run[verb]:
                return _PendingVerb(
                    verb=verb,
                    step_type="non_causative",
                    trigger_reason="match",
                    feature_count_causative=c_v,
                    feature_count_noncausative=n_v,
                )
            if self.batches_since_reservoir[verb] >= self.reservoir_stride:
                self.batches_since_reservoir[verb] = 0
                return _PendingVerb(
                    verb=verb,
                    step_type="non_causative",
                    trigger_reason="reservoir",
                    feature_count_causative=c_v,
                    feature_count_noncausative=n_v,
                )
        return None

    def on_train_batch_start(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        counts = self._count_features_per_verb(batch["index"])
        self._pending = []

        for verb in self.verbs:
            c_v, n_v = counts[verb]
            pending = self._decide_trigger(verb, c_v, n_v)
            if pending is None:
                continue

            df = self._run_eval(pl_module, self.dataloaders[verb])
            df = df.with_columns(
                pl.lit(verb).alias("verb"),
                pl.lit(trainer.global_step).alias("step"),
                pl.lit(trainer.current_epoch).alias("epoch"),
                pl.lit("pre").alias("phase"),
                pl.lit(pending.step_type).alias("step_type"),
                pl.lit(pending.trigger_reason).alias("trigger_reason"),
                pl.lit(pending.feature_count_causative).cast(pl.Int32).alias("feature_count_causative"),
                pl.lit(pending.feature_count_noncausative).cast(pl.Int32).alias("feature_count_noncausative"),
            )
            self.results[verb].append(df)
            self._pending.append(pending)

    def on_train_batch_end(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        for pending in self._pending:
            verb = pending.verb
            df = self._run_eval(pl_module, self.dataloaders[verb])
            df = df.with_columns(
                pl.lit(verb).alias("verb"),
                pl.lit(trainer.global_step).alias("step"),
                pl.lit(trainer.current_epoch).alias("epoch"),
                pl.lit("post").alias("phase"),
                pl.lit(pending.step_type).alias("step_type"),
                pl.lit(pending.trigger_reason).alias("trigger_reason"),
                pl.lit(pending.feature_count_causative).cast(pl.Int32).alias("feature_count_causative"),
                pl.lit(pending.feature_count_noncausative).cast(pl.Int32).alias("feature_count_noncausative"),
            )
            self.results[verb].append(df)

            if pending.step_type == "causative":
                self.causative_evals_run[verb] += 1
            else:
                self.noncausative_evals_run[verb] += 1

        self._pending = []

    def _run_eval(
        self,
        pl_module: ptl.LightningModule,
        dataloader: TorchDataLoader[DataBatchOvergen],
    ) -> pl.DataFrame:
        pl_module.eval()
        outputs: list[pl.DataFrame] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
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
        out_dir = ensure_dir(Path(trainer.default_root_dir) / "eval")
        for verb, frames in self.results.items():
            if not frames:
                continue
            pl.concat(frames).write_parquet(out_dir / f"monitor_step_{verb}.parquet")
