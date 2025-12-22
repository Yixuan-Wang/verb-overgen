# /// script
# [tool.marimo]
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import transformers as hf
    import tokenizers as hftk
    return Path, hf, hftk


@app.cell
def _():
    from verb_overgen.train.dataloader import DatasetPyChildes
    return (DatasetPyChildes,)


@app.cell
def _(DatasetPyChildes, Path, hf, hftk):
    tokenizer = hf.PreTrainedTokenizerFast(
            tokenizer_object=hftk.Tokenizer.from_file("/u501/y4466wan/projects/2025/verb-overgen/data/tokenizer/1024-wordlevel/tokenizer.json"),
        )
    dataset = DatasetPyChildes(
        input_path=Path("/u501/y4466wan/projects/2025/verb-overgen/data/childes-gold/test-251010T0422/merged.test.parquet/pychildes-morphosyntax.20251024.parquet"),
        tokenizer=tokenizer,
    )
    return (dataset,)


@app.cell
def _(dataset):
    dataset[0]
    return


if __name__ == "__main__":
    app.run()
