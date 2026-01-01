"""Merge different SCAMP data into a single table."""

from collections.abc import Iterable

import polars as pl
from polars._typing import IntoExpr
from polars._utils.parse.expr import parse_into_expression, parse_into_list_of_expressions

if __name__ != "__main__":
    raise RuntimeError("This script is not intended to be imported as a module.")

# %%
dataset_roberta = pl.read_parquet("./data/scamp/scamp-plausible-roberta.parquet")
dataset_llm = pl.read_parquet("./data/scamp/scamp-plausible-llm-infer.parquet")

dataset_roberta = dataset_roberta.sort(
    ["file_name", "file_line_idx", "is_grammatical"], descending=[False, False, True]
)
dataset_llm = dataset_llm.sort(["file_name", "file_line_idx", "is_grammatical"], descending=[False, False, True])

# %%
dataset = dataset_roberta.join(
    dataset_llm.select(["file_name", "file_line_idx", "is_grammatical", "llm_infer"]),
    on=["file_name", "file_line_idx", "is_grammatical"],
    coalesce=True,
    how="left",
)

def into_expr(expr: IntoExpr) -> pl.Expr:
    return pl.Expr._from_pyexpr(parse_into_expression(expr))

def into_exprs(*exprs: IntoExpr | Iterable[IntoExpr]) -> list[pl.Expr]:
    return list(map(pl.Expr._from_pyexpr, parse_into_list_of_expressions(*exprs)))

# %%
def zip_pad(*exprs: IntoExpr | Iterable[IntoExpr], pad: IntoExpr = None, pad_left: bool = False) -> list[pl.Expr]:
    """Pad lists to the same length with the given pad value.

    Args:
        *exprs: Expressions to be zipped. Must be list expressions.
        pad: Value to fill within the padding. Defaults to `None`.
        pad_left: Whether to pad on the left side. Defaults to `False`.

    Returns:
        List of padded expressions.
    """
    exprs: list[pl.Expr] = into_exprs(*exprs)
    len_max = pl.max_horizontal([expr.list.len() for expr in exprs])

    if pad_left:
        padded_exprs = [
            pl.when(expr.list.len() < len_max)
            .then(
                into_expr(pad).repeat_by(len_max - expr.list.len()).list.concat(expr).alias(expr.meta.output_name())
            )
            .otherwise(expr)
            for expr in exprs
        ]
    else:
        padded_exprs = [
            pl.when(expr.list.len() < len_max)
            .then(expr.list.concat(into_expr(pad).repeat_by(len_max - expr.list.len())))
            .otherwise(expr)
            for expr in exprs
        ]
    return padded_exprs

def zip_truncate(*exprs: IntoExpr | Iterable[IntoExpr], truncate_left: bool = False) -> list[pl.Expr]:
    """Truncate lists to the same length.

    Args:
        *exprs: Expressions to be zipped.
        truncate_left: Whether to truncate from the left side. Defaults to `False`.

    Returns:
        List of truncated expressions.
    """
    exprs: list[pl.Expr] = into_exprs(*exprs)
    len_min = pl.min_horizontal([expr.list.len() for expr in exprs])

    if truncate_left:
        truncated_exprs = [
            pl.when(expr.list.len() > len_min).then(expr.list.slice(-len_min)).otherwise(expr) for expr in exprs
        ]
    else:
        truncated_exprs = [expr.list.slice(0, len_min) for expr in exprs]

    return truncated_exprs

# %%
def mismatch(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return (
        df.select(
            "file_name",
            "file_line_idx",
            *zip_pad(pl.col("true"), pl.col("false")),
        )
        .with_columns(pl.int_ranges(pl.col("true").list.len()).alias("idx"))
        .explode(["true", "false", "idx"])
        .group_by(["file_name", "file_line_idx"], maintain_order=True)
        .agg(
            pl.col("idx").filter(pl.col("true").ne(pl.col("false"))).alias(name),
        )
    )

dataset_mismatch_lemma = (
    dataset.select("file_name", "file_line_idx", "is_grammatical", pl.col("roberta").struct["lemma"])
    .pivot(on="is_grammatical", values="lemma")
    .pipe(mismatch, name="idx_lemma_mismatch")
)

dataset_mismatch_word = (
    dataset.select("file_name", "file_line_idx", "is_grammatical", pl.col("list_word"))
    .pivot(on="is_grammatical", values="list_word")
    .pipe(mismatch, name="idx_word_mismatch")
)

dataset_mismatch = dataset_mismatch_lemma.join(
    dataset_mismatch_word, on=["file_name", "file_line_idx"], coalesce=True,
)

# %%
dataset_this = dataset.join(dataset_mismatch, on=["file_name", "file_line_idx"], how="left", coalesce=True)

# %%
expr_is_lemma_ok = (pl.col("idx_lemma_mismatch") == pl.col("idx_word_mismatch")).and_(pl.col("idx_lemma_mismatch").list.len() > 0)
expr_is_llm_verdict = pl.col("llm_infer").struct["verdict"].eq(True)

dataset_this.with_columns(
    expr_is_lemma_ok.alias("is_lemma_ok"),
    expr_is_llm_verdict.alias("is_llm_verdict"),
).write_parquet("./data/scamp/scamp-postprocessed.parquet")
