from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, cast

from jaxtyping import Float, Int
import lightning as ptl
from lightning.pytorch.loggers import WandbLogger
import polars as pl
import torch
import torch.nn.functional as nnf
from torch import Tensor
import transformers as hf
from transformers.modeling_outputs import CausalLMOutput
import tokenizers as hftk
from typing_extensions import override

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes, DataBatchPyChildes
from verb_overgen.train.lib.dataloader_scamp import DatasetScamp, evaluate_scamp

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

# Actual max length is 132
MAX_LEN: int = 160

class VerbOvergenTrainModule(ptl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()

        self.model = hf.GPT2LMHeadModel(model_config)
        self.eval_outputs = []

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

    def training_step(self, batch: DataBatchPyChildes, batch_idx: int):
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
        loss: Float[Tensor, ""] = this.loss # pyright: ignore[reportAssignmentType]

        self.log("train/loss", loss)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.eval_outputs.clear()

    def validation_step(self, batch, batch_idx):
        result: list[Float[Tensor, "ragged_seq"]] = evaluate_scamp(
            model=self.model,
            batch=batch,
        )
        self.eval_outputs.append(pl.DataFrame({
            "index_data": batch["index_data"],
            "token_prob": result,
        }))

    def on_validation_epoch_end(self) -> None:
        # TODO: Aggregate results
        pass

if __name__ == "__main__":
    # FIXME(production): Enable wandb
    environ["WANDB_MODE"] = "disabled"

    tokenizer = hf.PreTrainedTokenizerFast(
        tokenizer_object=hftk.Tokenizer.from_file("./data/tokenizer/1024-wordlevel/tokenizer.json"),
    )
    dataset_train = DatasetPyChildes(
        input_path=Path("./data/pychildes/pychildes-morphosyntax_postprocess.20260102_20251024.parquet"),
        token_id_pad=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=256,
        shuffle=True,
        collate_fn=dataset_train.collate_fn,
    )

    dataset_eval = DatasetScamp(
        input_path="./data/scamp/scamp-postprocessed-tokenized.20260101.parquet",
        token_id_pad=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=512,
        shuffle=False,
        collate_fn=dataset_eval.collate_fn,
    )

    model_config = hf.GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=MAX_LEN,
        n_ctx=MAX_LEN,
        n_embd=768,
        n_layer=8,
        n_head=8,
        #n_inner=4 * 768,
        pad_token_id=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
        bos_token_id=tokenizer._tokenizer.token_to_id(BOS_TOKEN),
        eos_token_id=tokenizer._tokenizer.token_to_id(EOS_TOKEN),
    )

    train_module = VerbOvergenTrainModule(
        model_config=model_config,
    )

    logger = WandbLogger(
        project="25-verb-overgen",
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

        # FIXME(production): Debug settings
        log_every_n_steps=10,
        max_steps=100,
        val_check_interval=10,

        logger=logger,
        default_root_dir="./var/train/20251230"
    )

    trainer.fit(
        train_module,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )

    # FIXME(production): Save final checkpoint
    trainer.save_checkpoint(
        "./var/train/20251230/verb-overgen_{}_final.pth".format(wandb_run_name or "run"),
    )

    logger.experiment.finish()
    logger.finalize("success")
