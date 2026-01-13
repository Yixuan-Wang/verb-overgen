from __future__ import annotations

from typing import TypedDict
import polars as pl

class Parse(TypedDict):
    """
    A dependency parse, represented as a dictionary of adjacency lists.
    """

    rel: list[str]
    arc: list[int]
    word: list[str]

DtParse: pl.DataType = pl.Struct(
    {
        "rel": pl.List(pl.String),
        "arc": pl.List(pl.Int64),
        "word": pl.List(pl.String),
    }
)
"""Dtype for the corresponding `Parse` type."""
