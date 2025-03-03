import polars as pl
from rich.progress import Progress
from typing import TypeVar
from functools import wraps
from collections.abc import Callable


WORKLOAD = 0
"""The total workload to be processed."""

def register_workload_on_df(df: pl.DataFrame):
    """
    Register the workload as the number of rows in the DataFrame.

    Args:
        df: The polars DataFrame to be processed.

    Returns:
        The same DataFrame left unchanged.
    """
    global WORKLOAD
    WORKLOAD = df.select(pl.len()).item()
    return df


F = TypeVar("F", bound=Callable)
def report_progress(func: F, advance: int = 1, description: str | None = None) -> F:
    """
    Decorator to print a `rich` progress bar while the decorated function is being mapped over
    the registered workload.
    Each call will advance the progress bar.

    Args:
        func (Callable): The function to decorate.
        advance (int): The amount to advance the progress bar by. Default is 1.
        description (str | None): The description of the progress bar. Default is None.

    Returns:
        The decorated function.
    """
    global WORKLOAD

    progress = Progress().__enter__()
    progress_task = progress.add_task(description or "", total=WORKLOAD)

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        progress.update(progress_task, advance=advance)
        if progress.finished:
            progress.__exit__(None, None, None)
        return result

    return wrapper  # type: ignore