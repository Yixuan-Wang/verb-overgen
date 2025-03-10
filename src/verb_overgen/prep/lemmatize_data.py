import fire
import polars as pl

from verb_overgen.prep.lib.lexical import to_lemma

def main(path: str):
    dataset = pl.read_parquet(path)

    if not input(f"""\
Path = {path},
Height = {dataset.height},
Lemmatize the dataset? (y/n): """) == "y":
        exit(1)

    dataset = dataset.with_columns(
        pl.col("parse").struct.field("word").list.eval(pl.element().map_elements(to_lemma, return_dtype=pl.String)).alias("lemma")
    )

    dataset.write_parquet(path)    


if __name__ == "__main__":
    fire.Fire(main)
