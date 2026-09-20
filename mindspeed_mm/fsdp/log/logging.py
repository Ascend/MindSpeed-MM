"""Core unified logging module."""

import logging
import os
import sys
from typing import Any, Optional, Union


LIBRARY_LOGGER_NAME = "mindspeed_mm.fsdp"

# Libraries that attach their own handler and set ``propagate = False`` on their library
# root logger.  Their records never reach the root logger, so ``global_scope`` has to
# unplug that handler explicitly to pull them into the common output.
THIRD_PARTY_LOGGER_NAMES = (
    "transformers",
    "accelerate",
    "datasets",
    "torch",
    "torch.distributed",
)

DEFAULT_FORMAT = (
    "[Rank %(rank)s | Local Rank %(local_rank)s] "
    "%(asctime)s %(levelname)s "
    "[%(relativepath)s:%(lineno)d] => %(message)s"
)


# Only handlers created by this module are tracked and managed, keyed by logger name.
_managed_handlers: dict[str, list[logging.Handler]] = {}

# Handlers this module detached from third-party loggers, together with the propagation
# setting they had, so that a later ``global_scope=False`` can hand them back.
_detached_third_party_loggers: dict[str, tuple[list[logging.Handler], bool]] = {}

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
    for handler in _managed_handlers.pop(logger.name, []):
        logger.removeHandler(handler)
        handler.close()


def _attach_handlers(
    logger: logging.Logger,
    level: Union[str, int],
    formatter: logging.Formatter,
    log_file: Optional[str],
) -> None:
    """Replace this module's handlers on ``logger`` with a fresh stdout/file pair.

    The level is set on the handlers as well as on the logger so that records coming from
    a foreign logger with a lower level (e.g. a library running at DEBUG) are still
    filtered by the configured level.
    """
    _remove_managed_handlers(logger)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    _managed_handlers[logger.name] = handlers


def _route_third_party_loggers_to_root() -> None:
    """Send third-party logs to the root logger instead of their private handler.

    Libraries such as ``transformers`` attach their own stderr handler to their library
    root logger and set ``propagate = False`` on it, so configuring the root logger alone
    leaves their output untouched and unformatted.  Dropping that handler and re-enabling
    propagation lets their records reach the root handlers installed by ``init_logger``
    without being printed twice.
    """
    for name in THIRD_PARTY_LOGGER_NAMES:
        third_party_logger = logging.getLogger(name)
        if name not in _detached_third_party_loggers:
            _detached_third_party_loggers[name] = (
                list(third_party_logger.handlers),
                third_party_logger.propagate,
            )
        for handler in list(third_party_logger.handlers):
            third_party_logger.removeHandler(handler)
        third_party_logger.propagate = True


def _restore_third_party_loggers() -> None:
    """Give third-party loggers back the handler and propagation setting they had."""
    for name, (handlers, propagate) in _detached_third_party_loggers.items():
        third_party_logger = logging.getLogger(name)
        for handler in handlers:
            third_party_logger.addHandler(handler)
        third_party_logger.propagate = propagate

    _detached_third_party_loggers.clear()


def init_logger(
    level: Optional[Union[str, int]] = None,
    format: str = DEFAULT_FORMAT,
    log_file: Optional[str] = None,
    global_scope: bool = True,
) -> None:
    """Initialize logging system.

    This function is intended to be called once during program startup.
    Repeated calls are supported and replace only handlers created by
    this module.

    Args:
        level: Logging level name or integer. Defaults to ``INFO``.
        format: Logging format string.
        log_file: Optional path for an additional file handler.
        global_scope: Additionally install the handlers on the root logger and route
            ``THIRD_PARTY_LOGGER_NAMES`` into them, so every record of the process is
            formatted the same way.  Set to ``False`` to keep logging confined to the
            ``mindspeed_mm.fsdp`` namespace.
    """
    if level is None:
        level = logging.INFO

    formatter = DynamicRankFormatter(format)

    # The namespace logger keeps its own handlers and never leaks into the root logger, so
    # it stays correctly formatted even when the surrounding framework reconfigures the
    # root logger or calls ``logging.basicConfig``.
    logger = _get_library_logger()
    logger.setLevel(level)
    logger.propagate = False
    _attach_handlers(logger, level, formatter, log_file)

    root_logger = logging.getLogger()

    if not global_scope:
        _remove_managed_handlers(root_logger)
        _restore_third_party_loggers()
        return

    # The root logger collects everything that propagates: third-party libraries and any
    # module using a plain ``logging.getLogger(__name__)``.
    root_logger.setLevel(level)
    _attach_handlers(root_logger, level, formatter, log_file)

    _route_third_party_loggers_to_root()


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
