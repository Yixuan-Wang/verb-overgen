from venv import logger
from apfel.container.maybe import Maybe
import polars as pl
import rich
from rich.progress import Progress
from rich.status import Status
from rich.console import Console
from string import Template

import atexit
import sys
from typing import Any, TypeVar
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


IS_INTERACTIVE: bool = sys.stdin.isatty()

CONSOLE: Console = rich.get_console()
CONSOLE.stderr = True


def span(shorthand: str):
    """
    Record a span.
    If running in interactive mode, the span will be recorded as a progress bar.
    If running in non-interactive mode, the span will be recorded as a log message.
    """
    ...


STATUS = Maybe[Status].nothing()


@atexit.register
def clear_status():
    global STATUS
    STATUS.map(lambda s: s.stop())  # type: ignore


def interactive_span(shorthand: str):
    global CONSOLE, STATUS

    clear_status()
    CONSOLE.clear_live()
    STATUS.replace(
        CONSOLE.status(
            shorthand,
        ).__enter__()
    )

def non_interactive_span(shorthand: str):
    logger.info("Entering span: %s", shorthand)


if False:
    interactive_span.__doc__ = span.__doc__
    span = interactive_span
else:
    non_interactive_span.__doc__ = span.__doc__
    span = non_interactive_span

class ProgressBar:
    """
    Create a progress bar or log message depending on the environment.
    If running in interactive mode, the progress bar will be displayed.
    If running in non-interactive mode, the progress bar will be logged as a message.
    """
    def __init__(self,
        total: int,
        debounce: int | None = None,
        message: str = "Running task..."
    ):
        pass

if IS_INTERACTIVE:
    class ProgressBar:
        def __init__(self,
            total: int,
            debounce: int | None = None,
            message: str = "Running task...",
            extras: dict[str, Any] = {}
        ):
            return
            rich.get_console().clear_live()
            self.progress = Progress()
            self.extras = extras
            self.extras["total"] = total

            self.message = Template(message)
            message_0 = self.message.safe_substitute(self.extras,step=0, total=total)
            self.task = self.progress.add_task(message_0, total=total)
            self.progress.start()

        def update(self, advance: int = 1, extras: dict[str, Any] = {}):
            return
            self.extras.update(extras)
            message = self.message.safe_substitute(self.extras, step=self.progress.tasks[0].completed + advance)
            self.progress.update(self.task, advance=advance, description=message)

        def stop(self):
            return
            self.progress.stop()
else:
    class ProgressBar:
        def __init__(self,
            total: int,
            debounce: int | None = None,
            message: str = "Running task...",
            extras: dict[str, Any] = {}
        ):
            self.total = total
            self.debounce = debounce or 1
            self.debounce_step = 0
            self.message = Template(message)
            self.step = 0
            self.extras = extras
            self.extras["total"] = total

        def update(self, advance: int = 1, extras: dict[str, Any] = {}):
            self.step += advance
            self.debounce_step += advance
            self.extras.update(extras)

            if self.debounce_step < self.debounce:
                return
            
            message = self.message.safe_substitute(self.extras, step=self.step, total=self.total)
            logger.info(message, extra={"step": self.step, "total": self.total})
            self.debounce_step = 0

        def stop(self):
            pass
