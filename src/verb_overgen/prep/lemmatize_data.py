from pathlib import Path
import fire
import polars as pl

from verb_overgen.util.log_util import report_progress, register_workload_on_df
from verb_overgen.prep.lib.lexical import vec_to_lemma, fix_spacing

lemmatize_dtype = pl.Struct(
    {
        "lemma": pl.List(pl.String),
        "pos": pl.List(pl.String),
    }
)


def text_to_lemma(text: str) -> dict[str, list[str]]:
    """
    Split the pre-"tokenized" text (with whitespaces between words, and word--punctuation)
    into a list of lemmas and a list of parts-of-speech (POS) tags.

    Calls `vec_to_lemma` to do the lexical part.
    """
    text_fixed = fix_spacing(text)
    words = list(filter(lambda s: s != "", text_fixed.split(" ")))


    # Call the lemmatizer
    lemma = vec_to_lemma(words)
    return lemma

def main(path: str):
    dataset = pl.read_parquet(path)

    dataset_lemma = dataset.pipe(register_workload_on_df).with_columns(
        pl.col("text")
        .map_elements(report_progress(text_to_lemma), lemmatize_dtype)
        .struct.unnest(),
    )

    new_path = Path(path).with_suffix(
        ".{timestamp}.lemma.parquet".format(
            timestamp=__import__("datetime").datetime.now().strftime(r"%Y%m%dT%H%M%S")
        )
    )
    dataset_lemma.write_parquet(new_path)


if __name__ == "__main__":
    fire.Fire(main)
