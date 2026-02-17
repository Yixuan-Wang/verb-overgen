"""Extract features from PyChildes data and mark them within the dataset."""

from collections.abc import Iterable
import polars as pl

from verb_overgen.prep.lib.pattern_matching import extract_periphrastic_make
from verb_overgen.prep.lib.lexical import LIST_CAUSATIVE_VERBS
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
    if "featured_verbs" not in dataset.columns:
        dataset = dataset.with_columns(pl.lit([], pl.List(pl.String)).alias("featured_verbs"))

    return dataset

def mark_causative(
    *,
    features: pl.Expr,
    featured_verbs: pl.Expr,
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
    
    list_matched_candidate_causative = list_matched_causative_make.list.set_intersection(list_matched_verb_candidate)
    
    _case = (
        pl.when(list_matched_candidate_causative.list.len() > 0).then(pl.lit("causative"))
        .when(list_matched_verb_candidate.list.len() > 0).then(pl.lit("causative_entrenched"))
        .when((list_matched_causative_make.list.len() > 0).and_(list_matched_candidate_causative.list.len() == 0)).then(pl.lit("unattended_causative"))
        .otherwise(pl.lit("none"))
    )
    
    ret_features = (
        pl.when(_case == "causative").then(features.list.set_union([FEATURE_CAUSATIVE]))
        .when(_case == "causative_entrenched").then(features.list.set_union([FEATURE_CAUSATIVE_ENTRENCHED]))
        .when(_case == "unattended_causative").then(features.list.set_union(["unattended_causative"]))
        .otherwise(features)
        .alias(features.meta.output_name())
    )
    
    def prefix_causative(expr: pl.Expr) -> pl.Expr:
        return expr.list.eval(pl.concat_str(pl.lit("causative:"), pl.element()))
    
    ret_featured_verbs = (
        pl.when(_case == "causative").then(featured_verbs.list.set_union(prefix_causative(list_matched_candidate_causative)))
        .when(_case == "causative_entrenched").then(featured_verbs.list.set_union(prefix_causative(list_matched_verb_candidate)))
        .when(_case == "unattended_causative").then(featured_verbs.list.set_union(prefix_causative(list_matched_causative_make)))
        .otherwise(featured_verbs)
        .alias(featured_verbs.meta.output_name())
    )

    return ret_features, ret_featured_verbs

if __name__ == "__main__":
    df = load_data("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")
    df = df.with_columns(
        mark_causative(
            features=pl.col("features"),
            featured_verbs=pl.col("featured_verbs"),
            pos=pl.col("morph").struct["pos"],
            parse=pl.col("parse"),
        )
    )

    df.write_parquet("./data/pychildes/pychildes-morphosyntax_postprocess.20260212_20251024.parquet")
