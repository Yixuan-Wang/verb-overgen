from typing import cast

import polars as pl
from torch.utils.data import DataLoader
import torch
from loguru import logger

from tqdm import tqdm

from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.train.train import TrainingSession
from verb_overgen.train.lib.dataloader_pychildes import DatasetPyChildes, remove_preemption_on_verb, remove_entrenchment_on_verb
from verb_overgen.train.lib.utils import load_train_config


def calc_count_overlap(dataset: DatasetPyChildes, *, randgen: torch.Generator | None):
    """
    Count the number of preemption samples that overlap with other verb's entrenchment by accident.
    """
    df_causative_count = (
        dataset.dataframe.filter(pl.col("features").list.contains(FEATURE_CAUSATIVE))
        .select("featured_verbs")
        .explode("featured_verbs")
        .filter(pl.col("featured_verbs").struct["feature"] == FEATURE_CAUSATIVE)
        .select(
            pl.col("featured_verbs").struct["verb"].value_counts().struct.unnest()
        )
    )
    
    count_overlap = 0

    for verb, count in zip(df_causative_count["verb"], df_causative_count["count"]):
        seed = torch.randint(0, 1_000_000, (1,)).item() if randgen else None
        df_entrenched = dataset.dataframe.filter(
            pl.col("featured_verbs").list.eval(
                pl.element().struct["feature"].eq(FEATURE_CAUSATIVE_ENTRENCHED) & pl.element().struct["verb"].eq(verb)
            ).list.any(),
        )
        if len(df_entrenched) < count:
            logger.warning(f"Not enough entrenched (but not preemptive) samples for verb '{verb}' to match the count of preemptive samples. Requested {count}, but only {len(df_entrenched)} available. ")
            continue

        df_sample = df_entrenched.sample(count, seed=seed)  # type: ignore
        count_overlap += df_sample.filter(pl.col("features").list.contains(FEATURE_CAUSATIVE)).select(pl.len()).item()
    
    return count_overlap

@__import__("ipdb").iex
def main():
    config = load_train_config()
    session = TrainingSession(config)
    _ = session.setup_tokenizer()
    dataset_train = cast(DatasetPyChildes, session.setup_datasets()["train"])
    
    remove_entrenchment_on_verb(dataset_train, verb="laugh", randgen=torch.Generator().manual_seed(42))
        
    __import__("IPython").embed(using=False)

    

if __name__ == "__main__":
    main()
