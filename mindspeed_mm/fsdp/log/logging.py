"""Core unified logging module."""

import logging
import os
import sys
from typing import Any, Optional, Union


LIBRARY_LOGGER_NAME = "mindspeed_mm.fsdp"

DEFAULT_FORMAT = (
    "[Rank %(rank)s | Local Rank %(local_rank)s] "
    "%(asctime)s %(levelname)s "
    "[%(relativepath)s:%(lineno)d] => %(message)s"
)


# Only handlers created by this module are tracked and managed.
_managed_handlers: list[logging.Handler] = []

# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class RelativePathFormatter(logging.Formatter):
    """Format source path relative to the current working directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cwd = os.getcwd()

    def format(self, record: logging.LogRecord) -> str:
        record.relativepath = os.path.relpath(
            record.pathname,
            start=self._cwd,
        )
        return super().format(record)


class DynamicRankFormatter(RelativePathFormatter):
    """Add global/local rank information when formatting a log record."""

    def format(self, record: logging.LogRecord) -> str:
        record.rank = int(os.getenv("RANK", 0))
        record.local_rank = int(os.getenv("LOCAL_RANK", 0))
        return super().format(record)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def _get_library_logger() -> logging.Logger:
    return logging.getLogger(LIBRARY_LOGGER_NAME)


def _remove_managed_handlers(logger: logging.Logger) -> None:
    """Remove only handlers created by this module."""
    for handler in _managed_handlers:
        logger.removeHandler(handler)
        handler.close()

    _managed_handlers.clear()


def init_logger(
    level: Optional[Union[str, int]] = None,
    format: str = DEFAULT_FORMAT,
    log_file: Optional[str] = None,
) -> None:
    """Initialize logging system.

    This function is intended to be called once during program startup.
    Repeated calls are supported and replace only handlers created by
    this module.

    Args:
        level: Logging level name or integer. Defaults to ``INFO``.
        format: Logging format string.
        log_file: Optional path for an additional file handler.
    """
    if level is None:
        level = logging.INFO

    logger = _get_library_logger()

    # Only configure the root namespace.
    logger.setLevel(level)
    logger.propagate = False

    # Replace handlers created by this module only.
    _remove_managed_handlers(logger)

    formatter = DynamicRankFormatter(format)

    # stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    _managed_handlers.append(stdout_handler)

    # optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        _managed_handlers.append(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the MindSpeed-MM FSDP namespace.

    Examples:
        ``get_logger()``
            -> ``mindspeed_mm.fsdp``

        ``get_logger(__name__)``
            -> normally returns the module's existing logger.

        ``get_logger("data_utils")``
            -> ``mindspeed_mm.fsdp.data_utils``
    """
    if not name:
        return _get_library_logger()

    # __name__ inside MindSpeed-MM is already a fully qualified logging namespace.
    if (
        name == LIBRARY_LOGGER_NAME
        or name.startswith(f"{LIBRARY_LOGGER_NAME}.")
    ):
        return logging.getLogger(name)

    return logging.getLogger(f"{LIBRARY_LOGGER_NAME}.{name}")


def bump_stacklevel(kwargs: dict[str, Any]) -> None:
    """Make logging report the original caller's source location."""
    kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
