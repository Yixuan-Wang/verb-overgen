from itertools import islice
from .train_model import DATASET, Hyperparam, TOKENIZER
import ipdb
import polars as pl

if __name__ == "__main__":
    from verb_overgen.train.lib.dataload import Dataloader

    hyperparam = Hyperparam()
    dataloader = Dataloader.from_dataframe(
        DATASET(),
        split="train",
        batch_size=hyperparam.batch_size,
        bos_token=TOKENIZER().bos_token_id, # type: ignore
    )

    __import__("IPython").embed(using=False)
    
    islice(dataloader, 582)

    dataloader.dataset_src.select(
        pl.col("tokenized").gather(dataloader.dataset_batch["index_src"])
    )

    dataloader.dataset_batch.select(
        pl.col("index_src").map_elements(
            lambda i: __import__("polars").Series.map_elements(
                dataloader.dataset_src[i, "tokenized"],
                lambda x: len(x.input_ids),
                __import__("polars").UInt64,
            ).to_list(),
            pl.List(pl.UInt64),
        ).list.min()
    )