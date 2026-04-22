from typing import cast

import numpy as np
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


def sample(
    df: pl.LazyFrame,
    *,
    size: int | None = None,
    fraction: float | None = None,
    seed: int | None = None,
    row_index_name: str = "row_index",
) -> pl.LazyFrame:
    """Sample rows without replacement from a LazyFrame.

    Exactly one of `size` or `fraction` must be provided. The frame's height is
    probed eagerly (cheap for parquet scans — reads metadata only), so this
    function is not fully lazy; it returns a LazyFrame so callers can keep
    composing downstream ops.

    Args:
        df: The LazyFrame to sample from.
        size: Number of rows to sample.
        fraction: Fraction of rows to sample, in [0, 1].
        seed: Seed for numpy's default_rng.
        row_index_name: Temporary column name used for filtering. Must not
            collide with an existing column.

    Returns:
        A LazyFrame with exactly `size` sampled rows (or `int(height * fraction)`).

    Raises:
        ValueError: If neither or both of `size`/`fraction` are provided, or if
            `size` exceeds the frame's height.
    """
    if (size is None) == (fraction is None):
        raise ValueError("Exactly one of `size` or `fraction` must be specified.")

    height: int = cast(pl.DataFrame, df.select(pl.len()).collect()).item()
    if size is None:
        size = int(height * fraction) 

    if size > height:
        raise ValueError(f"Requested sample size {size} exceeds frame height {height}.")

    sampled_indices = np.random.default_rng(seed).choice(height, size=size, replace=False)

    return (
        df.with_row_index(row_index_name)
        .filter(pl.col(row_index_name).is_in(pl.Series(sampled_indices)))
        # `.head(size)` lets polars stop scanning once it has accumulated `size`
        # matches. Best case O(max_sampled_index), worst case O(height).
        .head(size)
        .drop(row_index_name)
    )
