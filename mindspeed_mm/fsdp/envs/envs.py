"""MindSpeed-MM FSDP2 environment-variable registrations."""

from collections.abc import Collection
from typing import Any

from .manager import (
    Converter,
    EnvironmentVariable,
    EnvironmentVariableManager,
    Validator,
    parse_exact_true,
)

_manager = EnvironmentVariableManager()


def _register(
    name: str,
    default: Any,
    converter: Converter[Any] = str,
    description: str = "",
    *,
    deprecated: bool = False,
    replacement: str | None = None,
    choices: Collection[Any] | None = None,
    validator: Validator | None = None,
) -> None:
    _manager.register(
        EnvironmentVariable(
            name=name,
            default=default,
            converter=converter,
            description=description,
            deprecated=deprecated,
            replacement=replacement,
            choices=choices,
            validator=validator,
        )
    )


# Launcher-provided distributed context.
_register(
    name="NON_MEGATRON",
    default=False,
    converter=parse_exact_true,
    description="Select the non-Megatron FSDP2 initialization path.",
)
_register(
    name="RANK",
    default=0,
    converter=int,
    description="Global rank assigned by the distributed launcher.",
)
_register(
    name="WORLD_SIZE",
    default=1,
    converter=int,
    description="Total process count assigned by the distributed launcher.",
)
_register(
    name="LOCAL_RANK",
    default=0,
    converter=int,
    description="Node-local rank assigned by the distributed launcher.",
)
_register(
    name="LOCAL_WORLD_SIZE",
    default=1,
    converter=int,
    description="Process count on the current node.",
)

# MindSpeed-MM FSDP2 behavior.
_register(
    name="HF_SAVE_WAIT_MODE",
    default="sleep",
    converter=str,
    description="Waiting strategy used while saving Hugging Face checkpoints.",
)
_register(
    name="MM_FORCE_EP_BALANCE",
    default=False,
    converter=bool,
    description="Force expert-parallel load balancing for debugging.",
)

_manager.validate_registry()
environment_variables = _manager.variables
