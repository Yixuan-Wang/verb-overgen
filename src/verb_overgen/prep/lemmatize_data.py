from pathlib import Path
import fire
import polars as pl

from verb_overgen.util.log_util import report_progress, register_workload_on_df
from verb_overgen.prep.lib.lexical import vec_to_lemma, fix_spacing, SpacyMetadata, run_spacy

lemmatize_dtype = pl.Struct(
    {
        "lemma": pl.List(pl.String),
        "pos": pl.List(pl.String),
    }
)

spacy_metadata_dtype = pl.Struct(
    {
        "lemma": pl.List(pl.String),
        "morph": pl.List(pl.String),
        "pos_ud": pl.List(pl.String),
        "pos_penn": pl.List(pl.String),
        "dep_arc": pl.List(pl.Int32),
        "dep_rel": pl.List(pl.String),
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

def text_to_spacy(text: str) -> SpacyMetadata:
    """
    Split the pre-"tokenized" text (with whitespaces between words, and word--punctuation)
    into a list of Spacy metadata.

    - `lemma`: The lemmatized form of each word.
    - `morph`: The morphological features of each word, using UD FEATS representation. See
       https://spacy.io/api/morphology#morphanalysis for details.
    - `pos_ud`: The coarse grained part-of-speech tag of each word.
    - `pos_penn`: The fine grained part-of-speech tag of each word.
    - `dep_arc`: The dependency arc (head index) of each word.
    - `dep_rel`: The dependency relation label of each word.

    Calls `run_spacy` to do the lexical part.
    """
    text_fixed = fix_spacing(text)
    words = list(filter(lambda s: s != "", text_fixed.split(" ")))

    # Call the lemmatizer
    spacy_meta = run_spacy(words)
    return spacy_meta

def main(path: str):
    # TODO: add spacy metadata extraction option

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
