from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from transformers import AutoProcessor

from mindspeed_mm.fsdp.utils.dtype import get_dtype

from .base import ModelAdapter


class Qwen3_5Adapter(ModelAdapter):
    """Adapts standalone multimodal messages to Qwen3.5 generation inputs."""

    def __init__(
        self,
        model,
        processor_path: str,
        device: torch.device,
        param_dtype: str = None,
        enable_thinking: bool = False,
    ):
        if not processor_path:
            raise ValueError("Qwen3.5 requires inference.processor_path or model.model_name_or_path")
        self.model = model
        self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        self.device = device
        self.dtype = get_dtype(param_dtype) if param_dtype else None
        self.enable_thinking = enable_thinking

    @staticmethod
    def _normalize_media_path(value: Any) -> str:
        path = str(value)
        if path.startswith(("http://", "https://", "file://", "data:")):
            return path
        return str(Path(path).expanduser().resolve())

    @classmethod
    def _as_content(cls, messages: List[dict]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for item in messages:
            item_type = item["type"]
            value = item["value"]
            if item_type == "text":
                content.append({"type": "text", "text": str(value)})
            elif item_type in {"image", "video"}:
                values = value if isinstance(value, (list, tuple)) else [value]
                for media in values:
                    content.append({"type": item_type, item_type: cls._normalize_media_path(media)})
            else:
                raise ValueError(f"Unsupported message type: {item_type}")
        return content

    def _move_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                dtype = self.dtype if self.dtype is not None and torch.is_floating_point(value) else None
                moved[key] = value.to(device=self.device, dtype=dtype)
            else:
                moved[key] = value
        return moved

    def preprocess(self, messages: List[dict]) -> Dict[str, Any]:
        conversation = [{"role": "user", "content": self._as_content(messages)}]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            return_dict=True,
            return_tensors="pt",
        )
        return self._move_inputs(inputs)

    def generate(self, inputs: Dict[str, Any], generation_config):
        kwargs = generation_config.to_generate_kwargs(
            distributed=dist.is_initialized() and dist.get_world_size() > 1
        )
        if self.model.generation_config.pad_token_id is None:
            kwargs["pad_token_id"] = self.model.generation_config.eos_token_id
        return self.model.generate(**inputs, **kwargs)

    def decode(self, inputs: Dict[str, Any], outputs: Any) -> str:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("Qwen3.5 processor output does not contain input_ids")
        if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
            raise TypeError("Qwen3.5 model.generate() must return a two-dimensional token tensor")
        if outputs.shape[0] != input_ids.shape[0]:
            raise ValueError("Qwen3.5 generated batch size does not match the input batch size")

        generated_ids = [output_ids[input_ids.shape[1]:] for output_ids in outputs]
        decoder = getattr(self.processor, "batch_decode", None) or self.processor.tokenizer.batch_decode
        return decoder(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
