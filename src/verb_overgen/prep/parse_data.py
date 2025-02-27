from collections.abc import Callable
from functools import partial, wraps
from os import environ
from typing import Any, TypeVar
import polars as pl
from rich.progress import Progress
from rich import print
from supar import Parser
import torch
import warnings

import verb_overgen.util.pl_util as pl_util

# Here we use the `biaffine-dep-en`, which is less accurate but faster than
# the roberta version `biaffine-dep-roberta-en`
# :see-also: https://github.com/yzhangcs/parser?tab=readme-ov-file#dependency-parsing
#
# Also the documentation is [outdated](https://github.com/yzhangcs/parser/issues/145)
# Check [`supar.NAME`] for real names.
PARSER = Parser.load("biaffine-dep-en")

DATA_PATH_BASE = environ["DATA_PATH_BASE"]
dataset = pl.read_parquet(f"{DATA_PATH_BASE}/babylm/aochildes.parquet")

BATCH_SIZE = 32


torch.cuda.set_device("cuda:0")

# Trap all warnings
warnings.filterwarnings("ignore")


parse_dtype = pl.Struct(
    {
        "rel": pl.List(pl.String),
        "arc": pl.List(pl.Int64),
        "word": pl.List(pl.String),
    }
)
def parse(input: pl.Series) -> list[dict[str, list[Any]]]:
    global PARSER

    input = input.to_list()  # type: ignore
    parsed = PARSER.predict(input, lang="en", prob=True, verbose=False)

    return [
        {
            "rel": parsed.rels[i],
            "arc": parsed.arcs[i],
            "word": parsed.words[i],
        }
        for i in range(len(parsed))
    ]

WORKLOAD = 0
def register_workload(df: pl.DataFrame):
    global WORKLOAD
    WORKLOAD = df.select(pl.len()).item()
    return df

F = TypeVar("F", bound=Callable)
def report_progress(func: F, advance: int = 1, description: str | None = None) -> F:
    global WORKLOAD

    progress = Progress().__enter__()
    progress_task = progress.add_task(description or "", total=WORKLOAD)

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        progress.update(progress_task, advance=advance)
        if progress.finished:
            progress.__exit__(None, None, None)
        return result

    return wrapper  # type: ignore

if __name__ == "__main__":
    dataset = dataset.with_row_index()

    MIN_MEANINGFUL_SENTENCE_LEN = 3
    dataset_parsed = (
        dataset.filter(
            pl.col("text").str.split(" ").list.len().ge(MIN_MEANINGFUL_SENTENCE_LEN)
        )
        .pipe(register_workload)
        .select(
            pl.col("index"),
            (
                pl_util.batch(pl.col("text"), BATCH_SIZE)
                .map_elements(
                    report_progress(
                        parse,
                        advance=BATCH_SIZE,
                        description="Parsing sentences",
                    ),
                    return_dtype=pl.List(parse_dtype),
                )
                .explode()
                .alias("parse")
            ),
        )
        .pipe(partial(dataset.join, on="index", how="left"))
    )

    dataset_parsed.write_parquet(f"{DATA_PATH_BASE}/babylm/aochildes_parsed.parquet")
