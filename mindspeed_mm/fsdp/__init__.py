from . import envs as envs
from .ops import apply_ops_patch

if envs.NON_MEGATRON:
    apply_ops_patch()
