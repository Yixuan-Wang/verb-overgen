from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning as ptl
import polars as pl
import tokenizers as hftk
import torch
import transformers as hf
from apfel import Lazy, todo
from hydra import compose, initialize_config_dir
from jaxtyping import Float, Int
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

# from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.common.tokenize import BOS_TOKEN, PAD_TOKEN

# from verb_overgen.train.lib.dataloader_scamp import DatasetScamp, EvalScampCallback
from verb_overgen.train.lib.dataloader_overgen import DatasetOvergen, EvalOvergenCallback, EvalOvergenMonitor
from verb_overgen.train.lib.dataloader_pychildes import AugmentationSource, DataBatchPyChildes, DatasetPyChildes, TrainTokenMonitor, remove_entrenchment, remove_entrenchment_on_verb, remove_preemption, remove_preemption_on_verb, remove_token
from verb_overgen.train.lib.utils import CacheCleanCallback
from verb_overgen.train.lib.validate import eval_input_token_prob, VERB_TARGET_5
from verb_overgen.util.misc_util import ensure_dir

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
    from wandb.sdk.wandb_run import Run as WandbRun

# Actual max length is 132
# Previous max length is 512
MAX_LEN: int = 512


class VerbOvergenTrainModule(ptl.LightningModule):
    def __init__(self, model_config, optimizer_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = hf.GPT2LMHeadModel(model_config)
        self.optimizer_config = optimizer_config

    @override
    def configure_optimizers(self):
        param_dict = {param_name: param for param_name, param in self.model.named_parameters() if param.requires_grad}

        # https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L263
        # Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param for name, param in param_dict.items() if param.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.optimizer_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.optimizer_config.lr,
            betas=tuple(self.optimizer_config.betas),
        )

        warmup_steps = self.optimizer_config.scheduler.warmup_steps
        scheduler: LRSchedulerConfigType = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.optimizer_config.scheduler.start_factor,
                end_factor=self.optimizer_config.scheduler.end_factor,
                total_iters=warmup_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }

        ret: OptimizerLRSchedulerConfig = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

        return ret

    @override
    def training_step(self, batch: DataBatchPyChildes, batch_idx: int):
        if batch["input_ids"].shape[0] == 0:
            self.logger.experiment.alert(  # type: ignore
                title="Skipped training step",
                text=f"Skipped step {batch_idx} due to empty input",
                level="warn",  # Options: "warn" or "error"
                wait_duration=300,
            )
            return self.skip_training_step()

        seq_length = batch["seq_length"]
        # self.log("train/seq_length_mean", seq_length.float().mean())
        # self.log("train/seq_length_std", seq_length.float().std())
        self.log("train/total_tokens", seq_length.float().sum(), reduce_fx="sum")

        input_ids: Int[Tensor, "batch max_len"] = batch["input_ids"]
        attention_mask: Int[Tensor, "batch max_len max_len"] = batch["attention_mask"]
        labels: Int[Tensor, "batch max_len"] = torch.where(
            cast(Tensor, input_ids == self.model.config.pad_token_id),
            -100,
            input_ids,
        )

        this: CausalLMOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # logits: Float[Tensor, "batch seq vocab"] = this.logits
        loss: Float[Tensor, ""] = this.loss  # pyright: ignore[reportAssignmentType]

        self.log("train/loss", loss)
        return loss

    def skip_training_step(self):
        """
        Skip a training step by returning a zero loss.
        """
        first_param = next(param for param in self.model.parameters() if param.requires_grad)
        dummy_loss = first_param.sum() * 0.0
        return dummy_loss

    @override
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step."""
        if self.trainer.global_step % self.trainer.log_every_n_steps != 0:  # ty: ignore[unresolved-attribute]
            return  # Only log at specified intervals

        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float("inf"),  # Just compute, don't clip
        )
        self.log("train/grad_norm", grad_norm)

    def eval_input_token_prob_step(self, batch):
        result: list[Float[Tensor, "ragged_seq"]] = eval_input_token_prob(
            model=self.model,
            input_ids=batch["input_ids"],
            seq_length=batch["seq_length"],
        )

        return pl.DataFrame(
            {
                **{k: v for k, v in batch.items() if not isinstance(v, Tensor)},
                "token_prob": pl.Series("token_prob", result, pl.List(pl.Float32)),
            }
        )

    @override
    def validation_step(self, batch, batch_idx):
        return self.eval_input_token_prob_step(batch)

    @override
    def test_step(self, batch, batch_idx):
        return self.eval_input_token_prob_step(batch)


@Lazy
def wandb_logger() -> WandbLogger:
    # Setup Weights & Biases logger
    wandb_logger = WandbLogger(
        project="25-verb-overgen",
        save_dir=ensure_dir("./logs"),  # Wandb will create subdirectory here
        log_model=False,
    )

    return wandb_logger


class TrainingSession:
    """Manages the training session lifecycle including data loading, model setup, and training execution."""

    cfg: DictConfig
    run_name: str

    tokenizer: hf.PreTrainedTokenizerFast
    token_id_pad: int

    dataset_train: DatasetPyChildes
    dataset_eval_overgen: DatasetOvergen
    dataset_eval_overgen_subset: DatasetOvergen
    dataloader_train: torch.utils.data.DataLoader
    dataloader_eval_overgen: torch.utils.data.DataLoader
    dataloader_eval_overgen_subset: torch.utils.data.DataLoader

    model_config: hf.PretrainedConfig
    train_module: VerbOvergenTrainModule

    logger_wandb: WandbLogger

    dir_run_root: Path
    dir_checkpoint: Path

    trainer: ptl.Trainer

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger.info(OmegaConf.to_yaml(self.cfg))

        # Set seed for reproducibility
        if OmegaConf.select(self.cfg, "seed", default=None) is not None:
            ptl.seed_everything(self.cfg.seed)

    def setup_tokenizer(self, *, tokenizer_path: str | None = None) -> hf.PreTrainedTokenizerFast:
        """Load and configure the tokenizer.

        Args:
            tokenizer_path (str | None): Optional path to tokenizer file. If None, uses self.cfg.tokenizer.

        Prerequisites:
            - self.cfg.tokenizer: Path to tokenizer file

        Sets:
            - self.tokenizer: PreTrainedTokenizerFast instance
            - self.token_id_pad: Pad token ID
        """
        tokenizer_path = str(self.cfg.tokenizer) if tokenizer_path is None else tokenizer_path
        self.tokenizer = hf.PreTrainedTokenizerFast(
            tokenizer_object=hftk.Tokenizer.from_file(tokenizer_path),
        )
        self.token_id_pad = self.tokenizer._tokenizer.token_to_id(PAD_TOKEN)

        return self.tokenizer

    def setup_datasets(
        self,
        *,
        train_data_path: str | None = None,
        test_data_path: str | None = None,
    ) -> dict[str, torch.utils.data.Dataset]:
        """Create training and eval_overgen datasets.

        Args:
            train_data_path (str | None): Optional path to training data. If None, uses self.cfg.data.train.path.
            test_data_path (str | None): Optional path to eval_overgen data. If None, uses self.cfg.data.eval_overgen.path.

        Prerequisites:
            - self.token_id_pad: Pad token ID from setup_tokenizer()
            - self.cfg.data.train: Path to training data
            - self.cfg.data.eval_overgen: Path to eval_overgen data

        Sets:
            - self.dataset_train: DatasetPyChildes instance
            - self.dataset_eval_overgen: DatasetOvergen instance
        """
        assert hasattr(self, "token_id_pad"), "Tokenizer must be setup first"

        train_data_path = str(self.cfg.data.train.path) if train_data_path is None else train_data_path
        test_data_path = str(self.cfg.data.eval_overgen.path) if test_data_path is None else test_data_path

        raw_aug_sources = OmegaConf.select(self.cfg, "data.train.augmentation_sources", default=[])
        augmentation_sources: list[AugmentationSource] = [
            AugmentationSource(
                path_main=Path(src["path_main"]),
                path_annotation=Path(src["path_annotation"]),
                **({} if (s := src.get("size")) is None else {"size": int(s)}),
            )
            for src in cast(list[dict[str, Any]], OmegaConf.to_container(raw_aug_sources, resolve=True))
        ]
        augmentation_seed: int | None = OmegaConf.select(self.cfg, "data.train.subset_seed", default=None)

        self.dataset_train = DatasetPyChildes.from_path(
            input_path_main=Path(self.cfg.data.train.path),
            input_path_annotation=Path(self.cfg.data.train.path_annotation),
            token_id_pad=self.token_id_pad,
            max_len=MAX_LEN,
            max_token_per_batch=self.cfg.data.train.max_token_per_batch,
            augmentation_sources=augmentation_sources,
            augmentation_seed=augmentation_seed,
        )

        if OmegaConf.select(self.cfg, "data.train.debug", default=False):
            self.dataset_train.debug_truncate(1000)

        subset = self.cfg.data.train.subset
        if augmentation_sources and subset != "full-augmented":
            raise ValueError(
                f"augmentation_sources are set but subset={subset!r}. "
                "Use subset=full-augmented when providing augmentation sources."
            )
        if subset == "full-augmented" and not augmentation_sources:
            raise ValueError("subset=full-augmented requires at least one augmentation source.")

        match subset:
            case "full":
                pass
            case "full-augmented":
                pass
            case "remove_preemption":
                remove_preemption(self.dataset_train)
            case "remove_entrenchment":
                randgen = torch.Generator()
                if (subset_seed := OmegaConf.select(self.cfg, "data.train.subset_seed", default=None)) is not None:
                    randgen.manual_seed(subset_seed)

                remove_entrenchment(self.dataset_train, randgen=randgen)
            case "remove_preemption_on_verb":
                subset_verb = self.cfg.data.train.subset_verb_of_interest
                remove_preemption_on_verb(self.dataset_train, verb=subset_verb)
            case "remove_entrenchment_on_verb":
                subset_verb = self.cfg.data.train.subset_verb_of_interest
                randgen = torch.Generator()
                if (subset_seed := OmegaConf.select(self.cfg, "data.train.subset_seed", default=None)) is not None:
                    randgen.manual_seed(subset_seed)
                remove_entrenchment_on_verb(self.dataset_train, verb=subset_verb, randgen=randgen)
            case "remove_token_make":
                _sample_frac = OmegaConf.select(self.cfg, "data.train.subset_frac", default=1.)
                remove_token(self.dataset_train, token_id=self.tokenizer._tokenizer.token_to_id("make"), sample_frac=_sample_frac)
            case "none":
                logger.warning("No filtering logic applied to training dataset")

        self.dataset_eval_overgen = DatasetOvergen.from_path(
            input_path=self.cfg.data.eval_overgen.path,
            token_id_pad=self.token_id_pad,
        )
        
        tokens_to_watch = OmegaConf.select(self.cfg, "trace.tokens_to_watch", default=[])

        self.dataset_eval_overgen_subset = self.dataset_eval_overgen.create_subset_verb(
            tokens_to_watch,
        )

        return {
            "train": self.dataset_train,
            "eval_overgen": self.dataset_eval_overgen,
            "eval_overgen_subset": self.dataset_eval_overgen_subset,
        }

    def setup_dataloaders(self) -> dict[str, torch.utils.data.DataLoader]:
        """Create training and eval_overgen dataloaders.

        Prerequisites:
            - self.dataset_train: Training dataset from setup_datasets()
            - self.dataset_eval_overgen: eval_overgen dataset from setup_datasets()
            - self.cfg.data.train: Training data configuration
            - self.cfg.data.eval_overgen: eval_overgen data configuration

        Sets:
            - self.dataloader_train: DataLoader for training
            - self.dataloader_eval_overgen*: DataLoaders for evaluating overgen
        """
        assert (
            hasattr(self, "dataset_train")
            and hasattr(self, "dataset_eval_overgen")
            and hasattr(self, "dataset_eval_overgen_subset")
        ), "Datasets must be setup first"

        randgen_train_shuffling = torch.Generator()
        if (seed_train_shuffling := OmegaConf.select(self.cfg, "data.train.shuffle_seed", default=None)) is not None:
            randgen_train_shuffling.manual_seed(seed_train_shuffling)
        else:
           if hasattr(self, "logger_wandb") and self.logger_wandb.experiment:
                self.logger_wandb.experiment.config.update(
                    {
                        "train_shuffle_seed": randgen_train_shuffling.initial_seed(),
                    }
                )

        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_sampler=self.dataset_train.create_sampler(shuffle=randgen_train_shuffling),
            collate_fn=(
                self.dataset_train.collate_fn_without_filter
                if self.cfg.data.train.subset in ("full", "full-augmented")
                else self.dataset_train.collate_fn_with_filter
            ),
        )

        self.dataloader_eval_overgen = torch.utils.data.DataLoader(
            self.dataset_eval_overgen,
            batch_size=self.cfg.data.eval_overgen.batch_size,
            shuffle=False,
            collate_fn=self.dataset_eval_overgen.collate_fn,
        )

        self.dataloader_eval_overgen_subset = torch.utils.data.DataLoader(
            self.dataset_eval_overgen_subset,
            batch_size=self.cfg.data.eval_overgen.batch_size,
            shuffle=False,
            collate_fn=self.dataset_eval_overgen_subset.collate_fn,
        )

        ret = {
            "train": self.dataloader_train,
            "eval_overgen": self.dataloader_eval_overgen,
            "eval_overgen_subset": self.dataloader_eval_overgen_subset,
        }

        return ret

    def setup_model(self) -> torch.nn.Module:
        """Create model configuration and training module.

        Prerequisites:
            - self.tokenizer: Tokenizer, from setup_tokenizer()
            - self.cfg.model: Model configuration
            - self.cfg.optimizer: Optimizer configuration

        Sets:
            - self.model_config: GPT2Config instance
            - self.train_module: VerbOvergenTrainModule instance
        """
        assert hasattr(self, "tokenizer"), "Tokenizer must be setup first"

        self.model_config = hf.GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            n_positions=self.cfg.model.n_positions,
            n_ctx=self.cfg.model.n_ctx,
            n_embd=self.cfg.model.n_embd,
            n_layer=self.cfg.model.n_layer,
            n_head=self.cfg.model.n_head,
            pad_token_id=self.tokenizer._tokenizer.token_to_id(PAD_TOKEN),
            bos_token_id=self.tokenizer._tokenizer.token_to_id(BOS_TOKEN),
            # Note on the EOS token:
            # In the new setting, the EOS token is no longer kept.
            eos_token_id=self.tokenizer._tokenizer.token_to_id(BOS_TOKEN),
        )

        self.train_module = VerbOvergenTrainModule(
            model_config=self.model_config,
            optimizer_config=self.cfg.optimizer,
        )

        return self.train_module.model

    def setup_logging(self) -> WandbRun | None:
        """Configure wandb logger and directories.

        Prerequisites:
            - self.cfg.tokenizer: Path to tokenizer (for logging)
            - self.cfg.data: Data configuration (for logging)
            - self.cfg.model: Model configuration (for logging)
            - self.cfg.optimizer: Optimizer configuration (for logging)
            - self.cfg.output.root_dir: Root directory for outputs

        Sets:
            - self.logger_wandb: WandbLogger instance
            - self.run_name: Experiment run name (from wandb or "debug")
            - self.dir_run_root: Root output directory for this run
            - self.dir_checkpoint: Checkpoint subdirectory
        """
        self.logger_wandb = wandb_logger()

        if self.logger_wandb.experiment:
            # Get tags from config, default to empty list if not specified
            tags = OmegaConf.select(self.cfg, "wandb.tags", default=[])
            self.logger_wandb.experiment.tags += tags

            self.logger_wandb.experiment.config.update(
                {
                    "path_tokenizer": self.cfg.tokenizer,
                    "data_config": OmegaConf.to_container(self.cfg.data, resolve=True),
                    "model_config": OmegaConf.to_container(self.cfg.model, resolve=True),
                    "optimizer_config": OmegaConf.to_container(self.cfg.optimizer, resolve=True),
                }
            )
            self.run_name = self.logger_wandb.experiment.name or "debug"
        else:
            self.run_name = "debug"

        self.dir_run_root = ensure_dir(Path(f"{self.cfg.output.root_dir}/{self.run_name}"))
        self.dir_checkpoint = ensure_dir(self.dir_run_root / "checkpoints")

        return self.logger_wandb.experiment if self.logger_wandb.experiment else None

    def setup_trainer(self) -> ptl.Trainer:
        """Create PyTorch Lightning trainer with callbacks.

        Prerequisites:
            - self.logger_wandb: From setup_logging()
            - self.run_name: From setup_logging()
            - self.dir_checkpoint: From setup_logging()
            - self.dir_run_root: From setup_logging()
            - self.cfg.trainer: Trainer configuration

        Sets:
            - self.trainer: PyTorch Lightning Trainer instance
        """
        assert hasattr(self, "logger_wandb") and hasattr(self, "run_name"), "Logging must be setup first"
        assert hasattr(self, "dir_checkpoint") and hasattr(self, "dir_run_root"), "Directories must be setup first"

        callback_checkpoint = ModelCheckpoint(
            dirpath=str(self.dir_checkpoint),
            filename=f"{self.run_name}-{{epoch:02d}}",
            save_last="link",
            every_n_epochs=self.cfg.trainer.checkpoint.every_n_epochs,
            save_top_k=self.cfg.trainer.checkpoint.save_top_k,
        )

        tokens_to_watch = OmegaConf.select(self.cfg, "trace.tokens_to_watch", default=[])
        if isinstance(tokens_to_watch, str):
            raise TypeError(f"trace.tokens_to_watch must be a list, got string: {tokens_to_watch!r}")
        tokens_to_watch = list(tokens_to_watch)

        train_token_monitor = TrainTokenMonitor(
            tokens_to_watch=tokens_to_watch,
            tokenizer=self.tokenizer._tokenizer,
            dataset=self.dataset_train,
        )
        
        callbacks = [
            EvalOvergenCallback(),
            EvalOvergenMonitor(
                dataloader=self.dataloader_eval_overgen_subset,
                token_id_bos=self.tokenizer._tokenizer.token_to_id(BOS_TOKEN),
                run_every_n_steps=self.cfg.trainer.log_every_n_steps,
                primary_verb=OmegaConf.select(self.cfg, "data.train.subset_verb_of_interest") or (tokens_to_watch[0] if tokens_to_watch else None),
            ),
            callback_checkpoint,
            train_token_monitor,
            CacheCleanCallback(),
            LearningRateMonitor(),
        ]

        trainer_kwargs: dict[str, Any] = {
            "callbacks": callbacks,
            "accelerator": self.cfg.trainer.accelerator,
            "devices": self.cfg.trainer.devices,
            "max_epochs": self.cfg.trainer.max_epochs,
            "log_every_n_steps": self.cfg.trainer.log_every_n_steps,
            "val_check_interval": self.cfg.trainer.val_check_interval,
            "gradient_clip_val": self.cfg.trainer.gradient_clip_val,
            "logger": self.logger_wandb,
            "default_root_dir": self.dir_run_root,
        }

        self.trainer = ptl.Trainer(**trainer_kwargs)
        return self.trainer

    def train(self) -> None:
        """Execute the training loop.

        Prerequisites:
            - self.trainer: From setup_trainer()
            - self.train_module: From setup_model()
            - self.dataloader_train: From setup_dataloaders()
            - self.dataloader_eval_overgen: From setup_dataloaders()
        """
        assert hasattr(self, "trainer") and hasattr(self, "train_module"), "Trainer and module must be setup first"
        assert hasattr(self, "dataloader_train") and hasattr(self, "dataloader_eval_overgen"), (
            "Dataloaders must be setup first"
        )

        self.trainer.fit(
            self.train_module,
            train_dataloaders=self.dataloader_train,
            val_dataloaders=self.dataloader_eval_overgen,
        )

    def finalize(self) -> None:
        """Clean up and finalize logging.

        Prerequisites:
            - self.logger_wandb: From setup_logging()
        """
        assert hasattr(self, "logger_wandb"), "Logger must be setup first"

        self.logger_wandb.experiment.finish()
        self.logger_wandb.finalize("success")

    def setup(self) -> None:
        """Run all setup steps in order."""
        self.setup_logging()
        self.setup_tokenizer()
        self.setup_datasets()
        self.setup_dataloaders()
        self.setup_model()

        initial_weights = OmegaConf.select(self.cfg, "experiment.initial_weights", default=None)
        match initial_weights:
            case "use_pretrained":
                self.replace_weights_with_pretrained()
            case "use_pretrained_frozen":
                pretrained_params = self.replace_weights_with_pretrained()
                self.freeze_params(pretrained_params)

        self.setup_trainer()

    def run(self) -> None:
        self.train()
        self.finalize()

    def replace_weights_with_pretrained(self) -> set[str]:
        pretrained_model = hf.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

        pretrained_state_dict = pretrained_model.state_dict()
        target_model_state_dict = self.train_module.model.state_dict()

        # Filter out pretrained weights that are not in the target model
        # or have a shape mismatch (e.g., embedding layers).
        weights_to_load = {}
        for name, param in pretrained_state_dict.items():
            if name in target_model_state_dict:
                if param.shape == target_model_state_dict[name].shape:
                    weights_to_load[name] = param
                else:
                    logger.warning(
                        f"Skipping parameter '{name}' due to shape mismatch: target {target_model_state_dict[name].shape}, source {param.shape}"
                    )
            else:
                logger.warning(f"Skipping parameter '{name}' as it is not found in the target model.")

        # Load the filtered pretrained weights into the target model
        self.train_module.model.load_state_dict(weights_to_load, strict=False)

        logger.info(f"Successfully loaded {len(weights_to_load)} tensors from the pretrained model.")
        return set(weights_to_load.keys())

    def freeze_params(self, param_names: Collection[str]) -> None:
        """Freeze the specified model parameters.

        Args:
            param_names: Set of parameter names to freeze
        """
        frozen_count = 0
        for name, param in self.train_module.model.named_parameters():
            if name in param_names:
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"Frozen {frozen_count} parameters out of {len(param_names)} specified.")


def main(cfg: DictConfig):
    session = TrainingSession(cfg)
    session.setup()
    session.run()


if __name__ == "__main__":
    import sys
    from importlib.resources import files as resources

    config_dir = cast(Path, resources("verb_overgen.train.configs"))

    # Initialize Hydra with compose API
    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="train"):
        cfg = compose(config_name="train", overrides=sys.argv[1:])
        OmegaConf.resolve(cfg)
        main(cfg)
