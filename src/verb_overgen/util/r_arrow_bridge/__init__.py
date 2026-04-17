"""Python–R interop bridge via Apache Arrow.

Wraps an R source string in an isolated R environment and exposes four
call methods that pass a Polars or PyArrow table to a canonical `entry`
function defined in the R source:

- `RScript.call_table()` — table in, same-type table out (Arrow/dplyr
  lazy queries and tibbles are accepted and auto-converted).
- `RScript.call_scalar()` — table in, `pyarrow.Scalar` out
  (`arrow::Array`, `arrow::Scalar`, and length-1 base R atomics accepted).
- `RScript.call_void()` — table in, side effects only, `None` out.
- `RScript.call_value()` — table in, unboxed Python value out via rpy2's
  default conversion (length-1 R vector → scalar, longer → list, NULL → None).
  Requires base R return types; Arrow-backed column operations return Arrow
  objects — coerce with `as.numeric()` / `as.character()` when needed.

Every R source must define an entry-point function named `entry`:

    entry <- function(df) {
        # df is an arrow::Table
        df |> dplyr::mutate(z = x + 1)
    }

Each `RScript` instance evaluates its source into a fresh R environment,
so multiple instances with overlapping helper names do not interfere. `library()`
calls mutate the global R search path and are not thread-safe.

Dependencies (`rpy2`, `rpy2-arrow`, R `arrow >= 12.0.0`) are provided
by the outer project environment.

Example:

    import polars as pl

    script = RScript(\"\"\"
    library(arrow); library(dplyr)
    entry <- function(df) df |> mutate(z = x + 1)
    \"\"\")

    df = pl.DataFrame({"x": [1, 2, 3]})
    out: pl.DataFrame = script.call_table(df)
"""

from .bridge import RExecutionError, RScript

__all__ = ["RScript", "RExecutionError"]
