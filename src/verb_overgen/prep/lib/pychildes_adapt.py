"""Pipeline for adapting the parsed PyChildes data to a format suitable for training.

This module should be imported instead of run as a script.
"""

import polars as pl

PYCHILDES_LEMMA = pl.col("morph").struct.field("lemma")

def filter_valid_lemmas(lemma_expr: pl.Expr) -> pl.Expr:
    """Filter out valid lemmas from the given expression.

    Args:
        lemma_expr (pl.Expr): A `list[str]` Expression representing lemmas.

    Returns:
        pl.Expr: A `bool` expression indicating valid lemmas.
    """
    return lemma_expr.is_not_null().and_(lemma_expr.list.len() > 0)

def get_sentence(lemma_expr: pl.Expr) -> pl.Expr:
    """Get the sentence from the given lemma expression.

    Args:
        lemma_expr (pl.Expr): A `list[str]` Expression representing lemmas.
    
    Returns:
        pl.Expr: A `str` expression representing the sentence.
    """
    return lemma_expr.list.join(" ").str.replace(r"_", " ")

