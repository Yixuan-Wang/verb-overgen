from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, cast

import lightning as ptl
import polars as pl
import tokenizers as hftk
import torch
import torch.nn.functional as nnf
import transformers as hf
from jaxtyping import Float, Int
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override


from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from verb_overgen.train.lib.dataloader_pychildes import DataBatchPyChildes, DatasetPyChildes
from verb_overgen.train.lib.dataloader_scamp import DatasetScamp, EvalScampCallback
from verb_overgen.train.lib.dataloader_testset import DatasetTestset, TestTestsetCallback
from verb_overgen.train.lib.validate import eval_input_token_prob
from verb_overgen.util.misc_util import ensure_dir

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

# Actual max length is 132
MAX_LEN: int = 160


class VerbOvergenTrainModule(ptl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()

        self.model = hf.GPT2LMHeadModel(model_config)

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.0, betas=(0.9, 0.95))

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


if __name__ == "__main__":
    # environ["WANDB_MODE"] = "disabled"
    ptl.seed_everything(42)

    PATH_TRAIN = "./data/pychildes/pychildes-morphosyntax_postprocess.20260102_20251024.parquet"
    PATH_EVAL = "./data/scamp/scamp-postprocessed-tokenized.20260101.parquet"
    PATH_TEST = "./data/testset/output/testset_llm_gen.final_postprocessed_20260114T1637.parquet"
    PATH_TOKENIZER = "./data/tokenizer/1024-wordlevel/tokenizer.json"

    tokenizer = hf.PreTrainedTokenizerFast(
        tokenizer_object=hftk.Tokenizer.from_file(PATH_TOKENIZER),
    )
    token_id_pad = tokenizer._tokenizer.token_to_id(PAD_TOKEN)

    dataset_train = DatasetPyChildes(
        input_path=Path(PATH_TRAIN),
        token_id_pad=token_id_pad,
    )
    randgen_train_shuffling = torch.Generator()
    randgen_train_shuffling.manual_seed(42)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=256,
        shuffle=True,
        collate_fn=dataset_train.collate_fn,
        generator=randgen_train_shuffling,
    )

    dataset_val = DatasetScamp(
        input_path=PATH_EVAL,
        token_id_pad=token_id_pad,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=512,
        shuffle=False,
        collate_fn=dataset_val.collate_fn,
    )

    dataset_test = DatasetTestset(
        input_path=PATH_TEST,
        token_id_pad=token_id_pad,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=512, shuffle=False, collate_fn=dataset_test.collate_fn
    )

    model_config = hf.GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=MAX_LEN,
        n_ctx=MAX_LEN,
        n_embd=768,
        n_layer=8,
        n_head=8,
        # n_inner=4 * 768,
        pad_token_id=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
        bos_token_id=tokenizer._tokenizer.token_to_id(BOS_TOKEN),
        eos_token_id=tokenizer._tokenizer.token_to_id(EOS_TOKEN),
    )

    train_module = VerbOvergenTrainModule(
        model_config=model_config,
    )

    wandb_logger = WandbLogger(
        project="25-verb-overgen",
        save_dir=ensure_dir("./logs/wandb"),
        log_model=False,  # Use PyTorch Lightning to log models
    )
    wandb_logger.experiment.config.update(
        {
            "path_tokenizer": PATH_TOKENIZER,
            "path_dataset_train": PATH_TRAIN,
            "path_dataset_eval": PATH_EVAL,
            "path_dataset_test": PATH_TEST,
        }
    )

    wandb_run = wandb_logger.experiment
    run_name: str = wandb_run.name or "debug"
    PATH_ROOT = ensure_dir(f"./var/train/20251230/{run_name}")

    callback_checkpoint = ModelCheckpoint(
        dirpath=str(ensure_dir(Path(PATH_ROOT) / "checkpoints")),
        filename=f"{run_name}-{{epoch:02d}}",
        every_n_epochs=1,
        save_top_k=-1,
    )

    trainer = ptl.Trainer(
        callbacks=[EvalScampCallback(), TestTestsetCallback(), callback_checkpoint],
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        #
        max_epochs=3,
        log_every_n_steps=100,
        val_check_interval=0.25,
        #
        logger=wandb_logger,
        default_root_dir=PATH_ROOT,
    )

    trainer.fit(
        train_module,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )

    trainer.save_checkpoint(ensure_dir(Path(f"./var/train/20251230/{run_name}/checkpoints")) / f"{run_name}_final.pth")

    trainer.test(
        train_module,
        dataloaders=dataloader_test,
    )

    wandb_logger.experiment.finish()
    wandb_logger.finalize("success")
