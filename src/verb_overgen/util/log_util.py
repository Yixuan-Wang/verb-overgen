from contextlib import contextmanager
import datetime
import io
from apfel.container.maybe import Maybe
from loguru import logger
import polars as pl
import tqdm

from collections.abc import Callable, Generator
from functools import partial, wraps
from os import environ
import sys
from typing import Any, TypeVar, cast


IS_INTERACTIVE: bool = sys.stdin.isatty()
LOGURU_STDERR_HANDLER: Maybe[int] = Maybe.nothing()

if not IS_INTERACTIVE:
    if "TQDM_MININTERVAL" not in environ:
        environ["TQDM_MININTERVAL"] = "5"

def make_default_handler() -> Any:
    """
    Generate the configuration for the default logger.
    """
    return {
        "filter": {
            "": "INFO",
            "verb_overgen": "DEBUG" if IS_INTERACTIVE else "INFO",
            "transformers": "WARNING",
        }
    }

@contextmanager
def use_tqdm_logging_handler(pbar: tqdm.tqdm):
    global LOGURU_STDERR_HANDLER

    has_stderr_handler = LOGURU_STDERR_HANDLER.is_just()
    try:
        LOGURU_STDERR_HANDLER.take().map(logger.remove)
        handler = logger.add(
            partial(pbar.write, end=""),
            **make_default_handler() | {
                "colorize": IS_INTERACTIVE,
            },
        )
        LOGURU_STDERR_HANDLER.insert(handler)
        yield
    finally:
        LOGURU_STDERR_HANDLER.take().map(logger.remove)
        if has_stderr_handler:
            LOGURU_STDERR_HANDLER.insert(logger.add(sys.stderr, **make_default_handler()))

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
    Decorator to print a `tqdm` progress bar while the decorated function is being mapped over
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

    progress = tqdm.tqdm(None, total=WORKLOAD, desc=description)
    progress_ctx = use_tqdm_logging_handler(progress)
    progress_ctx.__enter__()

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        progress.update(advance)
        if progress.n >= WORKLOAD:
            progress.close()
            progress_ctx.__exit__(None, None, None)

        return result

    return wrapper  # type: ignore

def timedelta_to_iso8601(td: datetime.timedelta) -> str:
    """Convert a datetime.timedelta to an ISO 8601 duration string."""
    seconds = td.total_seconds()
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    
    with io.StringIO() as buf:
        buf.write("P")

        if days:
            buf.write(f"{int(days)}D")

        if hours or minutes or seconds:
            buf.write("T")
            if hours:
                buf.write(f"{int(hours)}H")
            if minutes:
                buf.write(f"{int(minutes)}M")
            if seconds:
                # Handle potential floating point seconds
                if seconds.is_integer():
                    buf.write(f"{int(seconds)}S")
                else:
                    buf.write(f"{seconds}S")

        result = buf.getvalue()

    # Edge case: zero duration
    if result == "P":
        result = "PT0S"

    return result

@contextmanager
def span(description: str):
    """
    Record a span.
    """

    logger_span = logger.opt(depth=2)
    logger_span.info(description)
    time_start = datetime.datetime.now()
    try:
        yield
    finally:
        time_end = datetime.datetime.now()
        time_diff = time_end - time_start

        logger_span.success(f"FINISHED {description} in {timedelta_to_iso8601(time_diff)}", extra={"elapsed": time_diff})

@contextmanager
def track(workload: Any) -> Generator[tqdm.tqdm, None, None]:
    global WORKLOAD

    if isinstance(workload, int):
        pass
    elif hasattr(workload, "__len__"):
        workload = len(workload)
    elif isinstance(WORKLOAD, int):
        workload = WORKLOAD
    else:
        workload = None
    workload: int | None
    
    pbar = tqdm.tqdm(total=workload)

    with (
        use_tqdm_logging_handler(pbar),
        pbar as pbar,
    ):
        yield pbar

def catch(func: F) -> F:
    if not IS_INTERACTIVE:
        return func
    else:
        try:
            import ipdb
        except ImportError:
            logger.warning("ipdb not found, will not catch exceptions")
            return func

        return ipdb.iex(func)
