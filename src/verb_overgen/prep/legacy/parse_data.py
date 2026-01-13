from functools import partial
from os import environ
from typing import Any
import polars as pl
from supar import Parser
import torch
import warnings

from verb_overgen.prep.lib.lexical import fix_spacing
from verb_overgen.prep.lib.syntactic import DtParse, Parse
from verb_overgen.util.log_util import register_workload_on_df, report_progress
import verb_overgen.util.pl_util as pl_util


# :see-also: https://github.com/yzhangcs/parser?tab=readme-ov-file#dependency-parsing
PARSER = Parser.load("dep-biaffine-roberta-en")

DATA_PATH_BASE = environ["DATA_PATH_BASE"]
dataset = pl.read_parquet(f"{DATA_PATH_BASE}/babylm/aochildes.parquet")

BATCH_SIZE = 64


torch.cuda.set_device("cuda:0")

# Trap all warnings
warnings.filterwarnings("ignore")

def parse(input: pl.Series) -> list[Parse]:
    """
    The parser is trained on PTB/train.conllx.
    """

    global PARSER

    input_fixed = [fix_spacing(s) for s in input.to_list()]    
    parsed: Any = PARSER.predict(input_fixed, lang="en", prob=True, verbose=False)

    # parsed.sentences is a list of such dictionaries.
    return [
        {
            "rel": sent.rels,
            "arc": sent.arcs,
            "word": sent.words,
        }
        for sent in parsed.sentences
    ]

if __name__ == "__main__":
    timestamp = __import__("datetime").datetime.now().strftime(r"%Y%m%dT%H%M%S")

    dataset = dataset.with_row_index()

    MIN_MEANINGFUL_SENTENCE_LEN = 3
    dataset_parsed = (
        dataset.filter(
            pl.col("text").str.split(" ").list.len().ge(MIN_MEANINGFUL_SENTENCE_LEN)
        )
        .pipe(register_workload_on_df)
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
                    return_dtype=pl.List(DtParse),
                )
                .explode()
                .alias("parse")
            ),
        )
        .pipe(partial(dataset.join, on="index", how="left"))
    )

    dataset_parsed.write_parquet(f"{DATA_PATH_BASE}/babylm/{timestamp}.aochildes_parsed.parquet")
