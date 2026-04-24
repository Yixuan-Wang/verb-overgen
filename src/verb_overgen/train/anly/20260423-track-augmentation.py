"""Analyze packed sequence statistics across the full dataset.

Records per packed sequence:
  - utterance count
  - PAD token count
  - whether it packs only 1 utterance
"""
import ipdb
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader

from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes
from verb_overgen.train.train import TrainingSession
from verb_overgen.train.lib.utils import load_train_config

PATH_MAIN = "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"
PATH_ANNOTATION = "./data/pychildes/pychildes-morphosyntax_annotation-20260225.20251024.parquet"
TOKEN_ID_PAD = 0
MAX_TOKEN_PER_BATCH = 8192
MAX_LEN = 512

config = load_train_config(overrides=["data.train.subset=full_augmented", "data.train.subset_verb_of_interest=laugh", "data=augment-laugh-causative"])
session = TrainingSession(config)

session.setup_tokenizer()
session.setup_datasets()
dataloaders = session.setup_dataloaders()

dataloader_train = dataloaders["train"]

for index, batch in enumerate(dataloader_train):
    print(f"Batch {index}.")
    if any(id_file_utterance.startswith("AUGMENTED") for id_file_utterance in batch["id_file_utterance"]):
        ipdb.set_trace()
