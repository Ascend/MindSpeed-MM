from .base import ModelAdapter
from .qwen3_5 import Qwen3_5Adapter


MODEL_ADAPTERS = {
    "qwen3_5": Qwen3_5Adapter,
    "qwen3_5_moe": Qwen3_5Adapter,
}


def build_adapter(name: str, **kwargs) -> ModelAdapter:
    adapter = MODEL_ADAPTERS.get(name, None)
    if adapter is None:
        raise ValueError(f"Unsupported adapter type: {name}")
    return adapter(**kwargs)
