import os
from dotenv import load_dotenv
from loguru import logger

import logging
import inspect
import sys
import warnings


# Load environment variables from .env file
load_dotenv()

# Reset the default logger
logger.remove()
logger.add(
    sys.stderr,
    filter={
        "": "INFO",
        "verb_overgen": "DEBUG" if sys.stdin.isatty() else "INFO",
        "transformers": "WARNING",
    }
)


# Sink warnings to loguru
showwarning_ = warnings.showwarning
def showwarning(message, *args, **kwargs):
    logger.opt(depth=2).warning(message)
warnings.showwarning = showwarning

# Disable tokenizers parallelism to avoid deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Intercept standard logging and redirect it to loguru.
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Configure logger according to debug mode
if sys.stdin.isatty():
    from rich.logging import RichHandler

    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )
    os.environ["LOGURU_LEVEL"] = "DEBUG"
    logger.debug("Running in interactive mode.")
else:
    logger.debug("Running in non-interactive mode.")

# Capture using ipdb
if sys.stdin.isatty():
    import ipdb
    ipdb.launch_ipdb_on_exception().__enter__()

# Add own `span` function to global scope
from verb_overgen.util.log_util import span # noqa: E402
import builtins # noqa: E402
builtins.span = span # type: ignore
