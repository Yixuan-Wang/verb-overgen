import polars as pl

def batch(expr: pl.Expr, batch_size: int):
    """
    Batch the expression into `pl.List` of `batch_size` length each.
    The last batch may have less than `batch_size` elements.

    Args:
        expr (pl.Expr): The expression to batch.
        batch_size (int): The batch size.

    Returns:
        pl.Expr: The batched expression.
    """
    return expr.implode().over(
        pl.int_range(pl.len()).floordiv(batch_size),
        mapping_strategy="explode",
    )
