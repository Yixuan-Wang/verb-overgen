from __future__ import annotations

from importlib.resources import files as _resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    import pandas as pd

import polars as pl
import pyarrow as pa
import rpy2.robjects as robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import r as _r
from rpy2_arrow.pyarrow_rarrow import (
    pyarrow_table_to_r_table,
    rarrow_to_py_array,
    rarrow_to_py_table,
)

ENTRY_POINT = "entry"
"""Canonical R entry-point name. Every RScript source must define `entry <- function(df) { ... }`."""

_WRAPPERS_R = """
.wrap_table_return <- function(x) {
    if (inherits(x, "Table")) x
    else if (inherits(x, "arrow_dplyr_query")) dplyr::compute(x)
    else if (inherits(x, "data.frame")) arrow::arrow_table(x)
    else stop(sprintf("entry() must return arrow::Table, arrow_dplyr_query, or data.frame, got: %s",
                      paste(class(x), collapse = "/")))
}
.wrap_scalar_return <- function(x) {
    if (inherits(x, "Array")) x
    else if (inherits(x, "Scalar")) arrow::Array$create(x$as_vector())
    else if (is.atomic(x) && length(x) == 1) arrow::Array$create(x)
    else stop(sprintf("entry() must return arrow::Array, arrow::Scalar, or length-1 atomic, got: %s (length %d)",
                      paste(class(x), collapse = "/"), length(x)))
}
"""

T = TypeVar("T", pa.Table, "pl.DataFrame", "pd.DataFrame")


def _unbox(r_obj: Any) -> Any:
    """Convert a basic rpy2 R object to a Python scalar, list, or None."""
    from rpy2.robjects import NULL as R_NULL

    if r_obj is R_NULL:
        return None
    # Basic atomic vectors (FloatVector, IntVector, StrVector, BoolVector)
    # expose __len__ and __getitem__ that already yield Python primitives.
    try:
        n = len(r_obj)
    except TypeError:
        return r_obj
    if n == 1:
        return r_obj[0]
    return list(r_obj)


class RExecutionError(Exception):
    """Raised when R-side evaluation fails; wraps rpy2's RRuntimeError."""


class RScript:
    """An R script compiled into an isolated environment, callable from Python via Arrow.

    The source must define a function named `entry` (see ENTRY_POINT):

    ```r
    entry <- function(df) {
        # df is an arrow::Table; return arrow::Table, data.frame, arrow::Array,
        # a length-1 atomic, or invisible(NULL) depending on which call_* method is used.
    }
    ```

    To work with a tibble inside ``entry``, collect the Arrow Table first:

    ```r
    entry <- function(df) {
        tbl <- dplyr::collect(df)   # arrow::Table -> tibble 
        df |> as_tibble()           # or alternatively
        # ... operate on tbl ...
        arrow::arrow_table(tbl)     # return as arrow::Table
    }
    ```

    Note:
        `library()` calls mutate the global R search path and are not thread-safe.
        Each `RScript` instance is isolated from others at the function-name level,
        but shared library state is global.
    """

    def __init__(self, source: str) -> None:
        """Compile R source into an isolated R environment.

        Args:
            source: R source code string. Must define a function named ``entry``.

        Raises:
            RExecutionError: If the source fails to parse/eval or does not define ``entry``.
        """
        self.source = source
        self._env = robjects.Environment()
        try:
            _r["eval"](_r["parse"](text=_WRAPPERS_R), envir=self._env)  # type: ignore
            _r["eval"](_r["parse"](text=source), envir=self._env)  # type: ignore
        except RRuntimeError as e:
            raise RExecutionError(self._format_error(e, "parse/eval")) from e
        if ENTRY_POINT not in self._env:
            raise RExecutionError(
                f"R source does not define entry point `{ENTRY_POINT}`"
            )

    @classmethod
    def from_resource(cls, package: str, filename: str) -> RScript:
        """Load R source from a package resource.

        Args:
            package: Dotted Python package containing the R file (e.g. ``"verb_overgen.analysis.r"``).
            filename: Resource filename (e.g. ``"glm_fit.R"``).

        Returns:
            A new RScript compiled from the resource file's contents.
        """
        source = _resources(package).joinpath(filename).read_text(encoding="utf-8")
        return cls(source)

    @classmethod
    def from_path(cls, path: str | Path) -> RScript:
        """Load R source from a filesystem path.

        Args:
            path: Path to a ``.R`` file.

        Returns:
            A new RScript compiled from the file's contents.
        """
        return cls(Path(path).read_text(encoding="utf-8"))

    @overload
    def call_table(self, df: pl.DataFrame, /) -> pl.DataFrame: ...
    @overload
    def call_table(self, df: pa.Table, /) -> pa.Table: ...
    @overload
    def call_table(self, df: pd.DataFrame, /) -> pd.DataFrame: ...
    def call_table(self, df: pl.DataFrame | pa.Table | pd.DataFrame, /) -> pl.DataFrame | pa.Table | pd.DataFrame:
        """Pass a table through ``entry`` and return the same type.

        The R ``entry`` function may return an ``arrow::Table`` or a ``data.frame``/tibble;
        both are accepted and converted to the caller's input type.

        Args:
            df: Input table (Polars or PyArrow).

        Returns:
            Transformed table in the same type as the input.

        Raises:
            RExecutionError: If R execution fails or the return type is unsupported.

        Example:
            >>> script = RScript('''
            ...   entry <- function(df) {
            ...     df |> dplyr::filter(score > 0.5)
            ...   }
            ... ''')
            >>> df = pl.DataFrame({"score": [0.3, 0.7, 0.9]})
            >>> script.call_table(df)
            shape: (2, 1)
            ┌───────┐
            │ score │
            │ ---   │
            │ f64   │
            ╞═══════╡
            │ 0.7   │
            │ 0.9   │
            └───────┘
        """
        r_result = self._invoke(self._to_r_table(df), wrapper=".wrap_table_return")
        pa_out = rarrow_to_py_table(r_result)
        if isinstance(df, pl.DataFrame):
            result = pl.from_arrow(pa_out)
            assert isinstance(result, pl.DataFrame)
            return result
        import pandas as pd  # noqa: PLC0415
        if isinstance(df, pd.DataFrame):
            return pa_out.to_pandas()
        return pa_out

    def call_scalar(self, df: pl.DataFrame | pa.Table | pd.DataFrame, /) -> pa.Scalar:
        """Pass a table to ``entry`` and return a scalar value.

        The R ``entry`` function must return an ``arrow::Array`` of length 1,
        or a length-1 base R atomic (numeric, integer, character, logical);
        the latter is auto-wrapped via ``arrow::Array$create``.

        Args:
            df: Input table (Polars or PyArrow).

        Returns:
            A PyArrow scalar extracted from the length-1 Arrow Array returned by R.

        Raises:
            RExecutionError: If R execution fails, the return type is wrong, or the Array length != 1.

        Example:
            >>> script = RScript('''
            ...   entry <- function(df) {
            ...     mean(df$score)
            ...   }
            ... ''')
            >>> df = pl.DataFrame({"score": [0.2, 0.4, 0.6]})
            >>> scalar = script.call_scalar(df)
            >>> scalar.as_py()
            0.4
        """
        r_result = self._invoke(self._to_r_table(df), wrapper=".wrap_scalar_return")
        arr = rarrow_to_py_array(r_result)
        if len(arr) != 1:
            raise RExecutionError(
                f"Scalar entry-point returned Array of length {len(arr)}, expected 1"
            )
        return arr[0]

    def call_value(self, df: pl.DataFrame | pa.Table | pd.DataFrame, /) -> Any:
        """Pass a table to ``entry`` and return an unboxed Python value.

        Uses rpy2's default conversion with automatic unboxing:

        - R ``NULL`` → ``None``
        - Length-1 R vector → Python scalar (``float``, ``int``, ``str``, ``bool``)
        - Longer R vector → ``list`` of Python scalars
        - Any other R object → raw rpy2 object

        Args:
            df: Input table (Polars or PyArrow).

        Returns:
            An unboxed Python value derived from the R return.

        Raises:
            RExecutionError: If R execution fails.

        Example:
            >>> script = RScript('''
            ...   entry <- function(df) {
            ...     nrow(df)
            ...   }
            ... ''')
            >>> df = pl.DataFrame({"x": [1, 2, 3]})
            >>> script.call_value(df)
            3
        """
        r_result = self._invoke(self._to_r_table(df), wrapper=None)
        return _unbox(r_result)

    def call_void(self, df: pl.DataFrame | pa.Table | pd.DataFrame, /) -> None:
        """Pass a table to ``entry`` for side effects, discarding the return value.

        The R ``entry`` function should return ``invisible(NULL)``.

        Args:
            df: Input table (Polars or PyArrow).

        Raises:
            RExecutionError: If R execution fails.

        Example:
            >>> script = RScript('''
            ...   entry <- function(df) {
            ...     write.csv(as.data.frame(df), "/tmp/out.csv", row.names = FALSE)
            ...     invisible(NULL)
            ...   }
            ... ''')
            >>> df = pl.DataFrame({"x": [1, 2, 3]})
            >>> script.call_void(df)  # writes /tmp/out.csv, returns None
        """
        self._invoke(self._to_r_table(df), wrapper=None)

    def _to_r_table(self, df: pl.DataFrame | pa.Table | pd.DataFrame):
        if isinstance(df, pl.DataFrame):
            return pyarrow_table_to_r_table(df.to_arrow())
        if isinstance(df, pa.Table):
            return pyarrow_table_to_r_table(df)
        import pandas as pd_  # noqa: PLC0415
        if isinstance(df, pd_.DataFrame):
            return pyarrow_table_to_r_table(pa.Table.from_pandas(df))
        raise TypeError(f"Expected pl.DataFrame, pa.Table, or pd.DataFrame, got {type(df)}")

    def _invoke(self, r_table, wrapper: str | None):
        try:
            entry = self._env[ENTRY_POINT]
            result = entry(r_table)
            if wrapper is not None:
                result = self._env[wrapper](result)
            return result
        except RRuntimeError as e:
            raise RExecutionError(self._format_error(e, ENTRY_POINT)) from e

    def _format_error(self, e: Exception, stage: str) -> str:
        head = self.source[:200].replace("\n", " \u21b5 ")
        return (
            f"R execution failed at stage `{stage}` "
            f"(source len={len(self.source)}): {e}\n"
            f"--- R source head ---\n{head}"
        )
