"""MindSpeed-MM FSDP2 environment-variable registrations."""

from collections.abc import Collection
from typing import Any

from .manager import (
    Converter,
    EnvironmentVariable,
    EnvironmentVariableManager,
    Validator,
    parse_bool,
    parse_exact_true,
    validate_non_negative,
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

# torch_npu CPU affinity (read-only reflection of the launch environment).
_register(
    name="CPU_AFFINITY_CONF",
    default="",
    converter=str,
    description=(
        "torch_npu CPU-affinity mode (e.g. '1' binds each rank to its "
        "card-local socket). Applied by torch_npu at the first backward; the "
        "training engine primes it at startup so early pinned allocations "
        "(swap arena) also land NUMA-local (see MM_PRIME_CPU_AFFINITY)."
    ),
)
_register(
    name="MM_PRIME_CPU_AFFINITY",
    default="auto",
    converter=str,
    choices=("auto", "0", "1"),
    description=(
        "Control the startup CPU-affinity priming backward independently of "
        "CPU_AFFINITY_CONF: 'auto' (default) primes only when CPU_AFFINITY_CONF "
        "is set; '0' never primes (A/B control or troubleshooting); '1' always "
        "primes, even without CPU_AFFINITY_CONF."
    ),
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
_register(
    name="MM_EP_BINCOUNT_DISPATCH",
    default=False,
    converter=bool,
    description=(
        "Use `torch.bincount` for count expert assignment when using expert-parallel training."
    ),
)
_register(
    name="MM_DETECT_ANOMALY",
    default=False,
    converter=bool,
    description=(
        "Enable PyTorch autograd anomaly detection for FSDP2 training debugging."
    ),
)
_register(
    name="TENSORBOARD_MAIN_RANK",
    default=0,
    converter=int,
    description="Global rank that writes TensorBoard scalars.",
    validator=validate_non_negative,
)
_register(
    name="GDN_SKIP_TRITON_AUTOTUNE",
    default=False,
    converter=parse_bool,
    description=(
        "Pin GDN triton kernels to the first autotune config and skip do_bench entirely "
        "(env-only switch; the GDN kernels never run the autotune benchmark sweep when set)."
    ),
)

_manager.validate_registry()
environment_variables = _manager.variables
