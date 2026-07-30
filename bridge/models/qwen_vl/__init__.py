__all__ = [
    "Qwen25VLBridge",
    "Qwen3VLMoEBridge",
    "Qwen3VLBridge",
]
from bridge.models.qwen_vl.qwen25_vl_bridge import Qwen25VLBridge
from bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLMoEBridge, Qwen3VLBridge
try:
    from bridge.models.qwen_vl.qwen3_5_bridge import Qwen3_5MoEBridge
except ImportError as exc:
    exc_name = getattr(exc, "name", "") or ""
    exc_msg = str(exc)

    known_optional_deps = (
        "mindspeed.te",
        "fla_npu",
    )

    if not (
        exc_name.startswith(known_optional_deps)
        or "mindspeed.te.pytorch.utils" in exc_msg
        or "get_tensor_model_parallel_group_if_none" in exc_msg
    ):
        raise

    Qwen3_5MoEBridge = None
