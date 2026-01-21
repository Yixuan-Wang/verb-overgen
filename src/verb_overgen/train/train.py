from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, cast

from apfel import Lazy
import lightning as ptl
import polars as pl
import tokenizers as hftk
import torch
import torch.nn.functional as nnf
import transformers as hf
from hydra import compose, initialize_config_dir
from jaxtyping import Float, Int
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from verb_overgen.common.features import FEATURE_CAUSATIVE
from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from verb_overgen.train.lib.dataloader_pychildes import DataBatchPyChildes, DatasetPyChildes
from verb_overgen.train.lib.dataloader_scamp import DatasetScamp, EvalScampCallback
from verb_overgen.train.lib.dataloader_testset import DatasetTestset, EvalTestsetCallback
from verb_overgen.train.lib.validate import eval_input_token_prob
from verb_overgen.util.misc_util import ensure_dir

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

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
            self.logger.experiment.alert(  # pyright:ignore
                title="Skipped training step",
                text=f"Skipped step {batch_idx} due to empty input",
                level="warn",  # Options: "warn" or "error"
                wait_duration=300,
            )
            return self.skip_training_step()

        seq_length = batch["seq_length"]
        self.log("train/seq_length_mean", seq_length.float().mean())
        self.log("train/seq_length_std", seq_length.float().std())
        self.log("train/total_tokens", seq_length.float().sum(), reduce_fx="sum")

        input_ids: Int[Tensor, "batch max_len"] = batch["input_ids"]
        attention_mask: Int[Tensor, "batch max_len"] = batch["attention_mask"]
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
        save_dir=ensure_dir("./logs"), # Wandb will create subdirectory here
        log_model=False,
    )

    return wandb_logger


def main(cfg: DictConfig):
    # Print configuration
    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    ptl.seed_everything(cfg.seed)

    # Load tokenizer
    tokenizer = hf.PreTrainedTokenizerFast(
        tokenizer_object=hftk.Tokenizer.from_file(cfg.tokenizer),
    )
    token_id_pad = tokenizer._tokenizer.token_to_id(PAD_TOKEN)

    expr_keep = pl.col("features").list.contains(FEATURE_CAUSATIVE).not_()

    dataset_train = DatasetPyChildes(
        input_path=Path(cfg.data.train.path),
        token_id_pad=token_id_pad,
        expr_keep=expr_keep,
    )
    randgen_train_shuffling = torch.Generator()
    if (seed_train_shuffling := OmegaConf.select(cfg, "data.train.shuffle_seed", default=None)) is not None:
        randgen_train_shuffling.manual_seed(seed_train_shuffling)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.data.train.batch_size,
        shuffle=cfg.data.train.shuffle,
        collate_fn=dataset_train.collate_fn,
        generator=randgen_train_shuffling,
    )

    # Create validation dataset and dataloader
    dataset_val = DatasetScamp(
        input_path=cfg.data.val.path,
        token_id_pad=token_id_pad,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.data.val.batch_size,
        shuffle=cfg.data.val.shuffle,
        collate_fn=dataset_val.collate_fn,
    )

    # Create test dataset and dataloader
    dataset_test = DatasetTestset(
        input_path=cfg.data.test.path,
        token_id_pad=token_id_pad,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.data.test.batch_size,
        shuffle=cfg.data.test.shuffle,
        collate_fn=dataset_test.collate_fn,
    )

    # Create model configuration
    model_config = hf.GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.model.n_positions,
        n_ctx=cfg.model.n_ctx,
        n_embd=cfg.model.n_embd,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        pad_token_id=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
        bos_token_id=tokenizer._tokenizer.token_to_id(BOS_TOKEN),
        eos_token_id=tokenizer._tokenizer.token_to_id(EOS_TOKEN),
    )

    # Create training module
    train_module = VerbOvergenTrainModule(
        model_config=model_config,
        optimizer_config=cfg.optimizer,
    )

    logger_wandb = wandb_logger()
    logger_wandb.experiment.config.update(
        {
            "path_tokenizer": cfg.tokenizer,
            "path_dataset_train": cfg.data.train.path,
            "path_dataset_eval": cfg.data.val.path,
            "path_dataset_test": cfg.data.test.path,
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
            "optimizer_config": OmegaConf.to_container(cfg.optimizer, resolve=True),
        }
    )
    run_name = logger_wandb.experiment.name or "debug"

    dir_run_root = ensure_dir(f"{cfg.output.root_dir}/{run_name}")
    dir_checkpoint = ensure_dir(Path(dir_run_root) / "checkpoints")

    # Setup model checkpoint callback
    callback_checkpoint = ModelCheckpoint(
        dirpath=str(dir_checkpoint),
        filename=f"{run_name}-{{epoch:02d}}",
        save_last="link",
        every_n_epochs=cfg.trainer.checkpoint.every_n_epochs,
        save_top_k=cfg.trainer.checkpoint.save_top_k,
    )

    # Create trainer with configuration
    trainer_kwargs = {
        "callbacks": [EvalScampCallback(), EvalTestsetCallback(), callback_checkpoint],
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "gradient_clip_val": cfg.trainer.gradient_clip_val,
        "max_epochs": cfg.trainer.max_epochs,
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "val_check_interval": cfg.trainer.val_check_interval,
        "logger": logger_wandb,
        "default_root_dir": dir_run_root,
    }

    trainer = ptl.Trainer(**trainer_kwargs)

    # Train the model
    trainer.fit(
        train_module,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )

    # Test the model
    trainer.test(
        train_module,
        dataloaders=dataloader_test,
    )

    # Finalize wandb
    logger_wandb.experiment.finish()
    logger_wandb.finalize("success")


if __name__ == "__main__":
    from importlib.resources import files as resources
    import sys
    
    config_dir = cast(Path, resources("verb_overgen.train.configs"))
    run_name: str = wandb_logger().experiment.name or "debug"

    # Initialize Hydra with compose API
    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name=run_name):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
        main(cfg)
