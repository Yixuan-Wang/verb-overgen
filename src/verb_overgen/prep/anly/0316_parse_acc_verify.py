# 0316: Parse Acc Verifier
# 
# Manually sample 100 parses and check if the dependency tree, lemmatization, and POS tags are correct.

import math
import ipdb
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from os import environ
import apfel as apfel
import statsmodels.api as sm
import rich
from rich.prompt import Confirm, Prompt
import json
import numpy as np
import numpy.random

from ..lib.graph import create_parse_graph, inject_pos_to_graph, draw_parse_graph
import warnings
import matplotlib

verification_time = __import__("datetime").datetime.now().strftime(r"%Y%m%dT%H%M%S")
output_path = f"/Users/tomyxw/Documents/Projects/work/verb-overgen/repo/src/verb_overgen/prep/asset/parse_verification/{verification_time}.jsonl"

# %%
dataset = pl.read_parquet(f"{environ['DATA_PATH_BASE']}/v2/babylm/0310/parsed-roberta.pos-lemma.parquet")

RANDOM_GEN = np.random.default_rng(42)
console = rich.get_console()

TOTAL_SAMPLES = 50

# Silence matplotlib GTK warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# For more specific GTK-related warnings
warnings.filterwarnings("ignore", message=".*GTK.*")


def interactive_loop():
    IDX_HAS_SAMPLED = set()
    REMAINING_SAMPLES = TOTAL_SAMPLES

    console.clear()

    while REMAINING_SAMPLES > 0:
        dataset_sample = (
            dataset
                .filter(
                    pl.col("parse").is_not_null()
                    & ~pl.col("index").is_in(IDX_HAS_SAMPLED)
                )
                .sample(n=REMAINING_SAMPLES, seed=RANDOM_GEN.integers(0, 2147483647))
            )


        for row in dataset_sample.iter_rows(named=True):
            

            console.print("#{sample_index} of {total_samples}\n[green]{index}[/]".format(sample_index = TOTAL_SAMPLES - REMAINING_SAMPLES, total_samples=TOTAL_SAMPLES, index=row["index"]))
            console.print("[blue]{sentence}[/]\n".format(sentence=row["text"]))

            is_good_english = Prompt.ask(
                "Is this example using good standard English?",
                choices=["y", "t", "g", "p", "m"],
                console=console,
            )

            if is_good_english != "y":
                match is_good_english:
                    case "g":
                        console.print("Skip for [red]grammatical error[/].")
                    case "t":
                        console.print("Skip for [red]tokenization error[/].")
                    case "p":
                        console.print("Skip for [red]punctuation error[/].")
                    case "m":
                        console.print("Skip for [red]paralinguistic[/].")
                yield {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "index": row["index"],
                    "sentence": row["text"],
                    "is_good_english": is_good_english,
                }
                continue

            REMAINING_SAMPLES -= 1
            IDX_HAS_SAMPLED.add(row["index"])

            console.print()
            words = row["parse"]["word"]
            lemma = row["lemma"]
            pos = row["pos"]
            for w, l, p in zip(words, lemma, pos):
                console.print(f"[green]{w:12}[/] [blue]{l:12}[/] [red]{p:12}[/]")
            console.print()

            is_pos_ok = Confirm.ask(
                "Is the POS tagging correct?",
                console=console,
            )
            is_lemma_ok = Confirm.ask(
                "Is the lemmatization correct?",
                console=console,
            )

            _ = Prompt.ask(
                "Press Enter to open the graph...",
                console=console,
                default="",
                show_default=False,
            )

            parse = row["parse"]
            parse_graph = create_parse_graph(parse)
            parse_graph = inject_pos_to_graph(parse_graph, pos)

            fig = draw_parse_graph(parse_graph)
            plt.show()

            while (is_parse_ok := Prompt.ask(
                "Is the dependency parse correct?",
                choices=["y", "p", "n", "a"],
                show_choices=True,
                console=console,
            )) == "a":
                fig = draw_parse_graph(parse_graph)
                plt.show()

            result = {
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "sentence": row["text"],
                "index": row["index"],
                "is_good_english": is_good_english,
                "is_pos_ok": is_pos_ok,
                "is_lemma_ok": is_lemma_ok,
                "is_parse_ok": is_parse_ok,
            }
            yield result
       
            plt.close(fig)
            del parse_graph

for result in interactive_loop():
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")


    console.print_json(json.dumps(result))
    _ = Prompt.ask(
        "Press Enter to continue",
        console=console,
    )
    console.clear()
