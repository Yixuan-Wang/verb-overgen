from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning as ptl
import polars as pl
import tokenizers as hftk
import torch
import transformers as hf
from apfel import Lazy
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
from verb_overgen.train.lib.dataloader_pychildes import DataBatchPyChildes, DatasetPyChildes

# from verb_overgen.train.lib.dataloader_scamp import DatasetScamp, EvalScampCallback
from verb_overgen.train.lib.dataloader_testset import DatasetTestset, EvalTestsetCallback
from verb_overgen.train.lib.utils import CacheCleanCallback
from verb_overgen.train.lib.validate import eval_input_token_prob
from verb_overgen.util.misc_util import ensure_dir

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
    from wandb.sdk.wandb_run import Run as WandbRun

# Actual max length is 132
MAX_LEN: int = 160


class VerbOvergenTrainModule(ptl.LightningModule):
    def __init__(self, model_config, optimizer_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = hf.GPT2LMHeadModel(model_config)
        self.optimizer_config = optimizer_config

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
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
        # attention_mask: Int[Tensor, "batch max_len max_len"] = batch["attention_mask"]
        # position_ids: Int[Tensor, "batch max_len"] = batch["position_ids"]
        labels: Int[Tensor, "batch max_len"] = torch.where(
            cast(Tensor, input_ids == self.model.config.pad_token_id),
            -100,
            input_ids,
        )

        this: CausalLMOutput = self.model(
            input_ids=input_ids,
            # position_ids=position_ids,
            # attention_mask=attention_mask,
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
            attention_mask=batch["attention_mask"],
            seq_length=batch["seq_length"],
        )

        return pl.DataFrame(
            {
                "index_sentence": batch["index_sentence"],
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
def wandb_logger():
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
    dataset_test: DatasetTestset
    dataloader_train: torch.utils.data.DataLoader
    dataloader_test: torch.utils.data.DataLoader

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
        """Create training and test datasets.

        Args:
            train_data_path (str | None): Optional path to training data. If None, uses self.cfg.data.train.path.
            test_data_path (str | None): Optional path to test data. If None, uses self.cfg.data.test.path.

        Prerequisites:
            - self.token_id_pad: Pad token ID from setup_tokenizer()
            - self.cfg.data.train: Path to training data
            - self.cfg.data.test: Path to test data

        Sets:
            - self.dataset_train: DatasetPyChildes instance
            - self.dataset_test: DatasetTestset instance
        """
        assert self.token_id_pad is not None, "Tokenizer must be setup first"

        train_data_path = str(self.cfg.data.train.path) if train_data_path is None else train_data_path
        test_data_path = str(self.cfg.data.test.path) if test_data_path is None else test_data_path

        self.dataset_train = DatasetPyChildes(
            input_path=Path(self.cfg.data.train.path),
            token_id_pad=self.token_id_pad,
            max_len=MAX_LEN,
            max_token_per_batch=self.cfg.data.train.max_token_per_batch,
        )

        self.dataset_test = DatasetTestset(
            input_path=self.cfg.data.test.path,
            token_id_pad=self.token_id_pad,
        )

        return {
            "train": self.dataset_train,
            "test": self.dataset_test,
        }

    def setup_dataloaders(self) -> dict[str, torch.utils.data.DataLoader]:
        """Create training and test dataloaders.

        Prerequisites:
            - self.dataset_train: Training dataset from setup_datasets()
            - self.dataset_test: Test dataset from setup_datasets()
            - self.cfg.data.train: Training data configuration
            - self.cfg.data.test: Test data configuration

        Sets:
            - self.dataloader_train: DataLoader for training
            - self.dataloader_test: DataLoader for testing
        """
        assert self.dataset_train is not None and self.dataset_test is not None, "Datasets must be setup first"

        randgen_train_shuffling = torch.Generator()
        if (seed_train_shuffling := OmegaConf.select(self.cfg, "data.train.shuffle_seed", default=None)) is not None:
            randgen_train_shuffling.manual_seed(seed_train_shuffling)

        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_sampler=self.dataset_train.create_sampler(shuffle=randgen_train_shuffling),
            collate_fn=self.dataset_train.collate_fn,
        )

        self.dataloader_test = torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.data.test.batch_size,
            shuffle=False,
            collate_fn=self.dataset_test.collate_fn,
        )

        return {
            "train": self.dataloader_train,
            "test": self.dataloader_test,
        }

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
        assert self.tokenizer is not None, "Tokenizer must be setup first"

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

            self.logger_wandb.experiment.config.update(
                {
                    "path_tokenizer": self.cfg.tokenizer,
                    "data_config": OmegaConf.to_container(self.cfg.data, resolve=True),
                    "model_config": OmegaConf.to_container(self.cfg.model, resolve=True),
                    "optimizer_config": OmegaConf.to_container(self.cfg.optimizer, resolve=True),
                    "tags": tags,
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
        assert self.logger_wandb is not None and self.run_name is not None, "Logging must be setup first"
        assert self.dir_checkpoint is not None and self.dir_run_root is not None, "Directories must be setup first"

        callback_checkpoint = ModelCheckpoint(
            dirpath=str(self.dir_checkpoint),
            filename=f"{self.run_name}-{{epoch:02d}}",
            save_last="link",
            every_n_epochs=self.cfg.trainer.checkpoint.every_n_epochs,
            save_top_k=self.cfg.trainer.checkpoint.save_top_k,
        )

        trainer_kwargs: dict[str, Any] = {
            "callbacks": [
                EvalTestsetCallback(),
                callback_checkpoint,
                CacheCleanCallback(),
                LearningRateMonitor(),
            ],
            "accelerator": self.cfg.trainer.accelerator,
            "devices": self.cfg.trainer.devices,
            "max_epochs": self.cfg.trainer.max_epochs,
            "log_every_n_steps": self.cfg.trainer.log_every_n_steps,
            "val_check_interval": self.cfg.trainer.val_check_interval,
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
            - self.dataloader_test: From setup_dataloaders()
        """
        assert self.trainer is not None and self.train_module is not None, "Trainer and module must be setup first"
        assert self.dataloader_train is not None and self.dataloader_test is not None, "Dataloaders must be setup first"

        self.trainer.fit(
            self.train_module,
            train_dataloaders=self.dataloader_train,
            val_dataloaders=self.dataloader_test,
        )

    def finalize(self) -> None:
        """Clean up and finalize logging.

        Prerequisites:
            - self.logger_wandb: From setup_logging()
        """
        assert self.logger_wandb is not None, "Logger must be setup first"

        self.logger_wandb.experiment.finish()
        self.logger_wandb.finalize("success")

    def setup(self) -> None:
        """Run all setup steps in order."""
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

        self.setup_logging()
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
        main(cfg)
