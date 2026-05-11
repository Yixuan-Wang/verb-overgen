from __future__ import annotations

from collections import Counter

import polars as pl
import torch
from torch.utils.data import DataLoader

from verb_overgen.train.lib.dataloader_pychildes import DataBatchPyChildes


def count_unigram_from_dataloader(
    dataloader: DataLoader[DataBatchPyChildes],
    *,
    progress: bool = False,
) -> pl.DataFrame:
    """Drain a DataLoader and return per-token-id unigram counts over `input_ids`.

    Counts every element of every batch's `input_ids` (padding included). The
    returned DataFrame has columns `token_id` (Int64) and `count` (Int64), sorted
    by descending count.
    """
    counter: Counter[int] = Counter()

    iterator = dataloader
    if progress:
        from tqdm import tqdm

        iterator = tqdm(dataloader)

    for batch in iterator:
        ids, counts = torch.unique(batch["input_ids"], return_counts=True)
        counter.update(dict(zip(ids.tolist(), counts.tolist(), strict=True)))

    return pl.DataFrame(
        {"token_id": list(counter.keys()), "count": list(counter.values())},
        schema={"token_id": pl.Int64, "count": pl.Int64},
    ).sort("token_id")


if __name__ == "__main__":
    from verb_overgen.train.lib.utils import load_train_config
    from verb_overgen.train.train import TrainingSession

    session = TrainingSession(load_train_config(overrides=[]))
    tokenizer = session.setup_tokenizer()
    dataset = session.setup_datasets()["train"]
    session.setup_dataloaders()

    df = count_unigram_from_dataloader(session.dataloader_train, progress=True)
    df = pl.select(
        pl.int_range(tokenizer.vocab_size).alias("token_id"),
    ).join(df, on="token_id", how="left").fill_null(0)

    df = df.sort("count", descending=True).head(20)
    print(df.head(20))
    print("total tokens:", df["count"].sum())

    __import__("IPython").embed(using=False)
