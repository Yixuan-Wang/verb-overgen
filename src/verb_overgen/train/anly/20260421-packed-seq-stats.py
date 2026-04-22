"""Analyze packed sequence statistics across the full dataset.

Records per packed sequence:
  - utterance count
  - PAD token count
  - whether it packs only 1 utterance
"""
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader

from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes

PATH_MAIN = "./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet"
PATH_ANNOTATION = "./data/pychildes/pychildes-morphosyntax_annotation-20260225.20251024.parquet"
TOKEN_ID_PAD = 0
MAX_TOKEN_PER_BATCH = 8192
MAX_LEN = 512

dataset = DatasetPyChildes.from_path(
    input_path_main=Path(PATH_MAIN),
    input_path_annotation=Path(PATH_ANNOTATION),
    token_id_pad=TOKEN_ID_PAD,
    max_token_per_batch=MAX_TOKEN_PER_BATCH,
    max_len=MAX_LEN,
)


def collate_with_utterance_count(batch):
    # Stash individual utterance lengths before collation packs them away
    utterance_lengths = [item["input_ids"].size(0) for item in batch]

    data = DatasetPyChildes.collate_utterances(
        batch,
        token_id_pad=TOKEN_ID_PAD,
        max_len=MAX_LEN,
    )

    # Recover per-packed-sequence utterance counts by greedily matching seq_length.
    # seq_length[i] == sum of utterance lengths in packed sequence i, so we
    # accumulate utterance lengths until the running total hits each seq_length.
    seq_lengths = data["seq_length"].tolist()
    utterance_counts: list[int] = []
    ptr = 0
    for sl in seq_lengths:
        acc, count = 0, 0
        while acc < sl:
            acc += utterance_lengths[ptr]
            ptr += 1
            count += 1
        utterance_counts.append(count)

    return data, utterance_counts


dataloader = DataLoader(
    dataset,
    batch_sampler=dataset.create_sampler(shuffle=None),
    collate_fn=collate_with_utterance_count,
)

all_utterance_counts: list[int] = []
all_pad_counts: list[int] = []

for batch, utterance_counts in dataloader:
    input_ids = batch["input_ids"]          # [n_packed, seq_padded]
    seq_lengths = batch["seq_length"]       # [n_packed]

    pad_counts = (input_ids.size(1) - seq_lengths).tolist()

    all_utterance_counts.extend(utterance_counts)
    all_pad_counts.extend(pad_counts)

n = len(all_utterance_counts)
n_single = sum(1 for c in all_utterance_counts if c == 1)
print(f"Total packed sequences : {n:,}")
print()
print("Utterances per packed sequence:")
print(f"  mean   {statistics.mean(all_utterance_counts):.2f}")
print(f"  median {statistics.median(all_utterance_counts)}")
print(f"  min    {min(all_utterance_counts)}")
print(f"  max    {max(all_utterance_counts)}")
print(f"  single-utterance sequences: {n_single:,} ({n_single / n:.1%})")
print()
print("PAD tokens per packed sequence:")
print(f"  mean   {statistics.mean(all_pad_counts):.2f}")
print(f"  median {statistics.median(all_pad_counts)}")
print(f"  min    {min(all_pad_counts)}")
print(f"  max    {max(all_pad_counts)}")
print(f"  total  {sum(all_pad_counts):,}")
