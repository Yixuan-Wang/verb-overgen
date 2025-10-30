import json
from pathlib import Path
from typing import Final
from loguru import logger
import polars as pl
from apfel.container.once import Lazy

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import os

from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN

@Lazy[pl.DataFrame]
def DATASET():
    dataset = pl.read_parquet(f"{os.environ['DATA_PATH_BASE']}/v2/0323/20250323T140740.aochildes_parsed.20250512T101328.lemma.parquet")
    return dataset

VOCAB_SIZE: Final = 8192

def train_bpe(dataset: pl.DataFrame) -> Tokenizer:
    corpus_tokenizer =  dataset.filter(pl.col("split") != "test").select(
        text=pl.col("lemma").list.join(" ")
    ).to_series()

    special_token = [
        PAD_TOKEN, # 0
        BOS_TOKEN, # 1
        EOS_TOKEN, # 2
        UNK_TOKEN, # 3
    ]

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN)) # type: ignore
    tokenizer.pre_tokenizer = Whitespace() # type: ignore
    trainer = BpeTrainer(special_tokens=special_token, vocab_size=VOCAB_SIZE) # type: ignore
    tokenizer.train_from_iterator(corpus_tokenizer, trainer=trainer) # type: ignore

    return tokenizer


# trainer = hftk_trainer.BpeTrainer(
#     dataset
# )

if __name__ == "__main__":
    now = __import__("datetime").datetime.now().strftime(r"%Y%m%dT%H%M%S")
    logger.info(f"Training tokenizer at {now}")

    tokenizer = train_bpe(DATASET())

    tokenizer_path = Path(f"{os.environ['DATA_PATH_BASE']}/tokenizer/0513/gpt2_aochildes_bpe_tokenizer--{now}")
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save((tokenizer_path / "tokenizer.json").as_posix())

    with open((tokenizer_path / "tokenizer_config.json").as_posix(), "w") as f:
        json.dump({
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
        }, f)
