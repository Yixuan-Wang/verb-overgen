from __future__ import annotations

from dataclasses import dataclass
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
    from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes, SamplerBin


@dataclass
class _PendingVerb:
    verb: str
    step_type: str
    trigger_reason: str
    feature_count_causative: int
    feature_count_noncausative: int
    batch_causative_total: int


class MonitorStepEffect(ptl.Callback):
    """Measure per-batch training-step effect on eval loss, separately per verb.

    A firing plan is precomputed once, before training starts, by replaying the
    (fixed, epoch-invariant) training sampler and deciding for every batch_idx
    which verbs should be probed:
    - causative: batch contains FEATURE_CAUSATIVE occurrences of the verb (always triggers)
    - non_causative (trigger_reason="entrench"): batch contains FEATURE_CAUSATIVE_ENTRENCHED
      occurrences of the verb, chosen from candidate batches so that the number of
      matched non-causative probes equals the number of causative probes for that
      verb, spread evenly across the epoch timeline.
    - periodic (trigger_reason="periodic"): a global heartbeat firing every
      `periodic_stride` batches, for every watched verb, evaluated post-step only.

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
        sampler: "SamplerBin",
        periodic_stride: int = 50,
    ) -> None:
        self.verbs = verbs
        self.dataset_dataframe = dataset_train.dataframe
        self.token_id_bos = token_id_bos
        self.periodic_stride = periodic_stride
        self._sampler = sampler

        self.dataloaders: dict[str, TorchDataLoader] = {}
        for verb in verbs:
            subset = dataset_eval_overgen.create_subset_verb([verb])
            self.dataloaders[verb] = torch.utils.data.DataLoader(
                subset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=subset.collate_fn,
            )

        self._plan: dict[int, list[_PendingVerb]] = {}
        self._active: list[_PendingVerb] = []
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

    def _count_causative_total(self, batch_index: list[int]) -> int:
        """Return the total number of FEATURE_CAUSATIVE occurrences across all verbs in the batch."""
        return (
            self.dataset_dataframe[batch_index]
            .lazy()
            .select("featured_verbs")
            .explode("featured_verbs")
            .filter(pl.col("featured_verbs").is_not_null())
            .filter(pl.col("featured_verbs").struct["feature"] == FEATURE_CAUSATIVE)
            .select(pl.len())
            .collect()
            .item()
        )

    def _build_plan(self) -> None:
        """Replay the training sampler once and precompute the firing plan for every batch_idx."""
        counts_per_batch: dict[int, dict[str, tuple[int, int]]] = {}
        causative_total_per_batch: dict[int, int] = {}
        causative_batches: dict[str, list[int]] = {v: [] for v in self.verbs}
        candidate_batches: dict[str, list[int]] = {v: [] for v in self.verbs}

        for batch_idx, batch_index in enumerate(self._sampler):
            counts = self._count_features_per_verb(batch_index)
            counts_per_batch[batch_idx] = counts
            causative_total_per_batch[batch_idx] = self._count_causative_total(batch_index)

            for verb in self.verbs:
                c_v, n_v = counts[verb]
                if c_v > 0:
                    causative_batches[verb].append(batch_idx)
                elif n_v > 0:
                    candidate_batches[verb].append(batch_idx)

        # 1. causative probes
        for verb in self.verbs:
            for batch_idx in causative_batches[verb]:
                c_v, n_v = counts_per_batch[batch_idx][verb]
                self._plan.setdefault(batch_idx, []).append(
                    _PendingVerb(
                        verb=verb,
                        step_type="causative",
                        trigger_reason="causative",
                        feature_count_causative=c_v,
                        feature_count_noncausative=n_v,
                        batch_causative_total=causative_total_per_batch[batch_idx],
                    )
                )

        # 2. entrench probes, stratified evenly across the epoch timeline
        for verb in self.verbs:
            cand = candidate_batches[verb]
            k = len(causative_batches[verb])
            selected: list[int] = []
            if k <= 0 or not cand:
                selected = []
            elif k >= len(cand):
                selected = list(cand)
            elif k == 1:
                selected = [cand[len(cand) // 2]]
            else:
                seen: set[int] = set()
                for i in range(k):
                    idx = cand[round(i * (len(cand) - 1) / (k - 1))]
                    if idx not in seen:
                        seen.add(idx)
                        selected.append(idx)

            for batch_idx in selected:
                c_v, n_v = counts_per_batch[batch_idx][verb]
                self._plan.setdefault(batch_idx, []).append(
                    _PendingVerb(
                        verb=verb,
                        step_type="non_causative",
                        trigger_reason="entrench",
                        feature_count_causative=c_v,
                        feature_count_noncausative=n_v,
                        batch_causative_total=causative_total_per_batch[batch_idx],
                    )
                )

        # 3. periodic heartbeat, one probe per verb every `periodic_stride` batches
        for batch_idx in counts_per_batch:
            if batch_idx % self.periodic_stride != 0:
                continue
            for verb in self.verbs:
                c_v, n_v = counts_per_batch[batch_idx][verb]
                self._plan.setdefault(batch_idx, []).append(
                    _PendingVerb(
                        verb=verb,
                        step_type="causative" if c_v > 0 else "non_causative",
                        trigger_reason="periodic",
                        feature_count_causative=c_v,
                        feature_count_noncausative=n_v,
                        batch_causative_total=causative_total_per_batch[batch_idx],
                    )
                )

    def on_fit_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> None:
        if self._plan:
            return
        self._build_plan()

    def on_train_batch_start(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._active = []

        for pending in self._plan.get(batch_idx, []):
            if pending.trigger_reason not in ("causative", "entrench"):
                continue

            verb = pending.verb
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
                pl.lit(pending.batch_causative_total).cast(pl.Int32).alias("batch_causative_total"),
            )
            self.results[verb].append(df)
            self._active.append(pending)

    def on_train_batch_end(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        for pending in self._active:
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
                pl.lit(pending.batch_causative_total).cast(pl.Int32).alias("batch_causative_total"),
            )
            self.results[verb].append(df)

        for pending in self._plan.get(batch_idx, []):
            if pending.trigger_reason != "periodic":
                continue

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
                pl.lit(pending.batch_causative_total).cast(pl.Int32).alias("batch_causative_total"),
            )
            self.results[verb].append(df)

        self._active = []

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
