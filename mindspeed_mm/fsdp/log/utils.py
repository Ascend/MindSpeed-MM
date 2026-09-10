import logging
import os
import sys
import threading
from typing import Any, Callable, Optional
import torch

from .logging import bump_stacklevel

def print_rank0(log_fn, message):
    """If distributed is initialized, print only on rank 0."""
    print_rank(log_fn, message, ranks=0)


def print_rank_last(log_fn, message):
    """If distributed is initialized, print only on last rank."""
    print_rank(log_fn, message, ranks=-1)


def print_rank_once(log_fn, message, ranks=0):
    """If distributed is initialized, print only on specified ranks, and only once per call-site."""
    frame = sys._getframe(1)  # Get the caller's frame
    return print_rank(log_fn, message, ranks=ranks, once=True, frame=frame)


def print_rank(log_fn, message, ranks=0, once=False, frame=None, **kwargs):
    """Print a log message on the specified ranks.

    Args:
        log_fn: Callable that emits the log message (e.g. a logger method)
        message: Log message to print
        ranks: Rank or ranks on which to print (default 0); negative values
            count from the end, e.g. -1 means the last rank
        once: If True, print only once per call site and rank
        frame: Caller frame used to identify the call site when once=True
        **kwargs: Extra arguments passed through to log_fn
    """
    # list-ify ranks for easier handling
    if ranks is None:
        ranks = []
    elif isinstance(ranks, int):
        ranks = [ranks]
    elif isinstance(ranks, tuple):
        ranks = list(ranks)

    # get the current rank and world size
    if not torch.distributed.is_initialized():
        rank_id = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    else:
        rank_id = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # support negative ranks, e.g., -1 means the last rank
    for i, rank in enumerate(ranks):
        if rank < 0:
            ranks[i] = world_size + rank

    if ranks is None or rank_id in ranks:
        if once and not _should_log_once(
            log_fn,
            rank_id,
            frame or sys._getframe(1),
            message,
        ):
            return
        # bump stacklevel if log_fn is a method of logging.Logger,
        # to ensure correct stacklevel in log output
        if isinstance(getattr(log_fn, "__self__", None), logging.Logger):
            bump_stacklevel(kwargs)
        log_fn(message, **kwargs)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_once_seen: set[tuple[Any, ...]] = set()
_once_lock = threading.Lock()


def _once_key(
    name: str,
    rank: Optional[int],
    frame: Any,
    message: str,
) -> tuple[Any, ...]:
    """Build a unique key for one call-site on one rank."""
    return (
        name,
        frame.f_code.co_filename,
        frame.f_lineno,
        rank,
        message,
    )


def _should_log_once(
    log_handler: Callable[..., None],
    rank: Optional[int],
    frame: Any,
    message: str,
) -> bool:
    """Return True only for the first call from a given call-site."""
    key = _once_key(
        name=getattr(log_handler, "__name__", str(log_handler)),
        rank=rank,
        frame=frame,
        message=message,
    )

    with _once_lock:
        if key in _once_seen:
            return False

        _once_seen.add(key)
    return True
