from itertools import islice
import os
from typing import Any
import ipdb
from loguru import logger
import numpy as np
import transformers as hf
import torch
import torch.nn.functional as F
from collections.abc import Iterator
from apfel.container.once import Lazy, Once
import polars as pl
from rich.progress import track
import wandb
from apfel.container.maybe import Maybe

from dataclasses import dataclass, field

import verb_overgen.train.lib.dataload as dataload
from verb_overgen.util import rand_util

HF_CHECKPOINT = "openai-community/gpt2"

MODEL: Lazy[hf.GPT2LMHeadModel] = Lazy(lambda: hf.GPT2LMHeadModel(hf.AutoConfig.from_pretrained(HF_CHECKPOINT)))
TOKENIZER: Lazy[hf.GPT2TokenizerFast] = Lazy(lambda: hf.AutoTokenizer.from_pretrained(HF_CHECKPOINT))  # type: ignore

@Lazy[Iterator[np.uint32]]
def GLOBAL_RAND_GEN():
    gen = rand_util.get_u32_rand_gen(None)
    np.random.seed(next(gen))
    torch.manual_seed(next(gen))
    import random
    random.seed(int(next(gen)))

    return gen

DEVICE: Once[torch.device] = Once()

@Lazy[pl.DataFrame]
def DATASET():
    dataset = pl.read_parquet(f"{os.environ['DATA_PATH_BASE']}/babylm/20250323T140740.aochildes_parsed.parquet")
    dataset = dataset.with_columns(
        tokenized=pl.col("text").map_elements(TOKENIZER(), return_dtype=pl.Object)  # type: ignore
    )

    return dataset

@dataclass(kw_only=True)
class Hyperparam:
    batch_size: int
    num_epochs: int
    learning_rate: float

@dataclass(kw_only=True)
class TrainState:
    epoch_count: int = 0
    batch_count: int = 0
    token_count: int = 0

    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Trainer():
    model: torch.nn.Module    
    hyper: Hyperparam
    dataloader: dataload.Dataloader
    optimizer: torch.optim.Optimizer = field(init=False)
    scheduler: torch.optim.lr_scheduler.LRScheduler = field(init=False)

    dataloader_dev: Maybe[dataload.Dataloader] = field(default_factory=Maybe.nothing)

    def __post_init__(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hyper.learning_rate,
            # TODO: add warmup
        )
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(
            self.optimizer,
            factor=1.0,
        )

    def on_train_start(self, state: TrainState):
        state.extras["wandb_run"] = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="tom-yx_wang",
            # Set the wandb project where this run will be logged.
            project="25-verb-overgen",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": hyperparam.learning_rate,
            },
        )

    def on_train_end(self, state: TrainState):
        pass

    def on_train_epoch_start(self, state: TrainState):
        pass

    def on_train_epoch_end(self, state: TrainState):
        pass

    def on_train_step_start(self, state: TrainState, data: dataload.Input):
        self.model.train()
        return data
    
    def on_train_step_end(self, state: TrainState, data: dataload.Input, output: float):
        state.token_count += data.input_ids.numel()

        if state.batch_count % 500 == 0:
            logger.info("devating model...")
            dev_loss = self.dev(Maybe.just(state))
        else:
            dev_loss = Maybe.nothing()

        run = state.extras["wandb_run"]
        run.log(
            {
                "epoch": state.epoch_count,
                "batch": state.batch_count,
                "loss": output,
                "token_count": state.token_count,
                **dev_loss.map(lambda x: {"dev_loss": x}).unwrap_or({}),
            }
        )

        logger.log(
            "DEBUG" if state.batch_count % 100 == 0 else "TRACE",
            f"Epoch {state.epoch_count}/{self.hyper.num_epochs} - Batch {state.batch_count}/{len(self.dataloader)} - Loss: {output:.4f}",
            extra={
                "epoch_count": state.epoch_count,
                "num_epochs": self.hyper.num_epochs,
                "batch_count": state.batch_count,
                "total_batches": len(self.dataloader),
                "loss": output,
            }
        )
  
        return output

    def train_step(self, state: TrainState, data: dataload.Input) -> float:
        # Forward pass
        output = self.model(input_ids=data.input_ids)
        logits: torch.FloatTensor = output.logits
        # shape: (batch_size, sequence_length, vocab_size)

        loss = F.cross_entropy(
            logits.flatten(0, 1).cpu(),
            data.target.view(-1),
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(self):
        state = TrainState()
        self.on_train_start(state)

        for epoch in range(self.hyper.num_epochs):
            state.epoch_count = epoch
            state.batch_count = 0
            self.on_train_epoch_start(state)
            for batch in track(self.dataloader):
                batch = self.on_train_step_start(state, batch)
                output = self.train_step(state, batch)
                _ = self.on_train_step_end(state, batch, output)
                state.batch_count += 1

            self.on_train_epoch_end(state)

        self.on_train_end(state)
        return

    @torch.no_grad
    def dev(self, state: Maybe[TrainState]) -> Maybe[float]:
        if self.dataloader_dev.is_nothing():
            return Maybe.nothing()

        dev_BATCH_SIZE = 16
        dev_loss = torch.empty(dev_BATCH_SIZE, device=torch.device("cpu"))

        self.model.eval()
        for i, batch in enumerate(
            islice(
                self.dataloader_dev.unwrap(), 16
            )
        ):
            output = self.model(input_ids=batch.input_ids)
            logits: torch.FloatTensor = output.logits
            loss = F.cross_entropy(
                logits.flatten(0, 1).cpu(),
                batch.target.view(-1),
            )

            dev_loss[i] = loss.item()

        return Maybe.just(dev_loss.mean().item())


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        from verb_overgen.train.lib.dataload import Dataloader

        DEVICE.set(torch.device("cuda").__enter__())

        span("load-data")
        hyperparam = Hyperparam(
            batch_size=32,
            num_epochs=5,
            learning_rate=5e-4,
        )
        dataloader = Dataloader.from_dataframe(
            DATASET().filter(dataload.split("train")),
            batch_size=hyperparam.batch_size,
            bos_token=TOKENIZER().bos_token_id, # type: ignore
            seed_global_shuffle=rand_util.get_u32_rand_gen(next(GLOBAL_RAND_GEN())),
            seed_batch_shuffle=rand_util.get_u32_rand_gen(next(GLOBAL_RAND_GEN())),
        )

        dataloader_dev = Dataloader.from_dataframe(
            DATASET().filter(dataload.split("dev")),
            batch_size=32,
            bos_token=TOKENIZER().bos_token_id, # type: ignore
            seed_global_shuffle=42,
            seed_batch_shuffle=42,
        )

        span("load-model")
        model = MODEL()

        span("load-trainer")
        trainer = Trainer(
            model=model,
            hyper=hyperparam,
            dataloader=dataloader,
            dataloader_dev=Maybe.just(dataloader_dev),
        )

        span("train")
        trainer.train()
