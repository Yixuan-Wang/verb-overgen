"""Extract features from PyChildes data and mark them within the dataset."""

from collections.abc import Iterable
import polars as pl

from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_make
from verb_overgen.prep.lib.lexical import BIDGOOD_VERBS
from verb_overgen.common.features import FEATURE_CAUSATIVE, FEATURE_CAUSATIVE_ENTRENCHED
from verb_overgen.prep.lib.pattern_matching import extract_verb


def load_data(file_path: str) -> pl.DataFrame:
    """Load data.

    Extra steps:
    - Create a `parse` column following the `[arc, rel, word]` format, if not present.
    """
    dataset = pl.read_parquet(file_path)

    if "parse" not in dataset.columns:
        dataset = dataset.with_columns(
            pl.struct(
                pl.col("syntax").struct.field("to_index").alias("arc"),
                pl.col("syntax").struct.field("rel").alias("rel"),
                pl.col("morph").struct.field("lemma").alias("word"),
            ).alias("parse")
        )

    if "features" not in dataset.columns:
        dataset = dataset.with_columns(pl.lit([], pl.List(pl.Categorical("features"))).alias("features"))

    return dataset


LIST_CAUSATIVE_VERBS = list(BIDGOOD_VERBS["intransitive_only"])

def mark_causative(
    features: pl.Expr,
    *,
    pos: pl.Expr,
    parse: pl.Expr,
    candidate_verbs: Iterable[str] | None = None,
) -> pl.Expr:
    """Flag presence of periphrastic causative constructions, defined in `lib.pattern_matching.extract_periphrastic_make`.

    Args:
        parse (pl.Expr): The parsed data. Should be evaluated to dtype `DtParse`.
        features (pl.Expr): The existing features list.
        candidate_verbs (Iterable[str] | None): The set of verbs to consider. If `None`, use `SET_CAUSATIVE_VERBS`, defined in `lib.lexical.BIDGOOD_VERBS` as intransitive-only verbs.

    Returns:
        pl.Expr: The updated features list.
    """

    list_verb = extract_verb(pos=pos, word=parse.struct["word"])
    list_matched_verb_candidate = list_verb.list.set_intersection(LIST_CAUSATIVE_VERBS if candidate_verbs is None else list(candidate_verbs))

    list_matched_causative_make = extract_periphrastic_make(parse=parse)
    
    list_matched_candidate_causative = list_matched_causative_make.list.set_intersection(list_verb)

    return (
        pl.when(list_matched_candidate_causative.list.len() > 0)
        .then(features.list.set_union([FEATURE_CAUSATIVE]))
        .when(list_matched_verb_candidate.list.len() > 0)
        .then(features.list.set_union([FEATURE_CAUSATIVE_ENTRENCHED]))
        .when((list_matched_causative_make.list.len() > 0).and_(list_matched_candidate_causative.list.len() == 0))
        .then(features.list.set_union(["unattended_causative"]))
        .otherwise(features)
        .alias(features.meta.output_name())
    )

if __name__ == "__main__":
    df = load_data("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")
    df = df.with_columns(
        pl.col("features").pipe(
            mark_causative,
            pos=pl.col("morph").struct["pos"],
            parse=pl.col("parse"),
        )
    )

    df.write_parquet("./data/pychildes/pychildes-morphosyntax_postprocess.20260102_20251024.parquet")
