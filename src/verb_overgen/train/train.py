from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float, Int
import lightning as ptl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor
import transformers as hf
from transformers.modeling_outputs import CausalLMOutput
import tokenizers as hftk
from typing_extensions import override

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from verb_overgen.train.dataloader import DatasetPyChildes, DataPyChildes

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

# Actual max length is 132
MAX_LEN: int = 160

class VerbOvergenTrainModule(ptl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()

        # self.model = hf.GPT2LMHeadModel(model_config)
        self.model = hf.OPTForCausalLM(model_config)

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0., betas=(0.9, 0.95))

        warmup_steps = 1000
        scheduler: LRSchedulerConfigType = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / warmup_steps,
                end_factor=1.0,
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
    
    def pad_to_max_len(self, batch: DataPyChildes) -> DataPyChildes:
        input_ids: Int[Tensor, "batch seq"] = batch["input_ids"]
        attention_mask: Int[Tensor, "batch seq"] = batch["attention_mask"]

        input_ids: Int[Tensor, "batch max_len"] = nnf.pad(
            input_ids,
            pad=(0, MAX_LEN - input_ids.size(1)),
            value=self.model.config.pad_token_id,
        )

        attention_mask: Int[Tensor, "batch max_len"] = nnf.pad(
            attention_mask,
            pad=(0, MAX_LEN - attention_mask.size(1)),
            value=0,
        )

        ret: DataPyChildes = {
            **batch,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return ret

    def training_step(self, batch: DataPyChildes, batch_idx: int):
        seq_length = batch["seq_length"]
        self.log("train/seq_length_mean", seq_length.float().mean())
        self.log("train/seq_length_std", seq_length.float().std())
        self.log("train/total_tokens", seq_length.float().sum(), reduce_fx="sum")

        batch = self.pad_to_max_len(batch)


        input_ids: Int[Tensor, "batch max_len"] = batch["input_ids"]
        attention_mask: Int[Tensor, "batch max_len"] = batch["attention_mask"]
        labels: Int[Tensor, "batch max_len"] = torch.where(
            input_ids == self.model.config.pad_token_id,
            -100,
            input_ids,
        )
        

        this: CausalLMOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # logits: Float[Tensor, "batch seq vocab"] = this.logits
        loss: Float[Tensor, ""] = this.loss # type: ignore

        self.log("train/loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = self.pad_to_max_len(batch)

        input_ids: Int[Tensor, "batch max_len"] = batch["input_ids"]
        attention_mask: Int[Tensor, "batch max_len"] = batch["attention_mask"]
        labels: Int[Tensor, "batch max_len"] = torch.where(
            input_ids == self.model.config.pad_token_id,
            -100,
            input_ids,
        )

        this: CausalLMOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # logits: Float[Tensor, "batch seq vocab"] = this.logits
        loss: Float[Tensor, ""] = this.loss # type: ignore
        self.log("val/loss", loss)

if __name__ == "__main__":
    tokenizer = hf.PreTrainedTokenizerFast(
        tokenizer_object=hftk.Tokenizer.from_file("/u501/y4466wan/projects/2025/verb-overgen/data/tokenizer/1024-wordlevel/tokenizer.json"),
    )
    dataset = DatasetPyChildes(
        input_path=Path("/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0422/merged.test.parquet/pychildes-morphosyntax.20251024.parquet"),
        tokenizer=tokenizer,
    )

    random_gen = torch.Generator().manual_seed(42)
    valid_size = 4096
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset,
        [len(dataset) - valid_size, valid_size],
        generator=random_gen,
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        # num_workers=15,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=128,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # model_config = hf.GPT2Config(
    #     vocab_size=tokenizer.vocab_size,
    #     n_positions=MAX_LEN,
    #     n_ctx=MAX_LEN,
    #     n_embd=768,
    #     n_layer=8,
    #     n_head=8,
    #     #n_inner=4 * 768,
    #     pad_token_id=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
    #     bos_token_id=tokenizer._tokenizer.token_to_id(BOS_TOKEN),
    #     eos_token_id=tokenizer._tokenizer.token_to_id(EOS_TOKEN),
    # )

    model_config = hf.OPTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        max_position_embeddings=MAX_LEN,
        num_hidden_layers=8,
        num_attention_heads=8,
        # ffn_dim = 4 * hidden_size,
        pad_token_id=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
        bos_token_id=tokenizer._tokenizer.token_to_id(BOS_TOKEN),
        eos_token_id=tokenizer._tokenizer.token_to_id(EOS_TOKEN),
    )

    train_module = VerbOvergenTrainModule(
        model_config=model_config,
    )

    logger = WandbLogger(
        project="25-verb-overgen",
        save_dir="data",
    )

    wandb_run = logger.experiment
    wandb_run_name: str = wandb_run.name

    # TODO: Enable checkpointing
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="/u501/y4466wan/projects/2025/verb-overgen/data/train/gpt2",
    #     filename="verb-overgen_gpt2-a6cbd4_{step}.ckpt",
    #     every_n_train_steps=10000,
    # )

    trainer = ptl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        val_check_interval=1000,
        logger=logger,
        default_root_dir="/u501/y4466wan/projects/2025/verb-overgen/data/train/opt"
    )

    trainer.fit(
        train_module,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )

    trainer.save_checkpoint(
        "/u501/y4466wan/projects/2025/verb-overgen/data/train/opt/verb-overgen-opt-a6cbd4_{}_final.pth".format(wandb_run_name or "run"),
    )

    logger.experiment.finish()
    logger.finalize("success")
