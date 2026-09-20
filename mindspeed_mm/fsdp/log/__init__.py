"""
Unified logging utilities for the MindSpeed-MM FSDP backend.

This package hosts parallel, independent modules:
- ``logging`` (planned by RFC #664): unified text logging.
- ``tensorboard``: TensorBoard scalar recording backend.
- ``metrics``: unified metrics entry point with pluggable handler backends.

The modules are independent of each other and can be enabled separately.
"""

from mindspeed_mm.fsdp.log.tensorboard import tb_writer
from mindspeed_mm.fsdp.log.metrics import metrics, TensorBoardHandler
from mindspeed_mm.fsdp.log.logging import init_logger, get_logger
from mindspeed_mm.fsdp.log.utils import print_rank


__all__ = [
    "tb_writer",
    "metrics",
    "TensorBoardHandler",
    "init_logger",
    "get_logger",
    "print_rank",
]
