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
from verb_overgen.util.log_util import make_default_handler, LOGURU_STDERR_HANDLER  # noqa: E402
stderr_handler = logger.add(
    sys.stderr,
    **make_default_handler(),
)
LOGURU_STDERR_HANDLER.insert(stderr_handler)

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
    logger.debug("Running in interactive mode.")
else:
    logger.debug("Running in non-interactive mode.")
