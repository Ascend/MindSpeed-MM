import importlib
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration


class ModelZoo:

    MODEL_MAPPINGS = {
        "qwen3_vl": Qwen3VLForConditionalGeneration,
        "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration
    }

    @staticmethod
    def build(config, transformer_config):
        architectures = getattr(transformer_config, "architectures", [])
        model_type = getattr(transformer_config, "model_type", None)

        model_cls = None
        if architectures:
            transformers_module = importlib.import_module("transformers")
            model_cls = getattr(transformers_module, architectures[0], None)
            if model_cls is not None:
                return model_cls

        if model_type:
            model_cls = ModelZoo.MODEL_MAPPINGS.get(model_type, None)
            if model_cls is not None:
                return model_cls

        model_id = getattr(config, "model_id", None)
        if model_id:
            model_cls = ModelZoo.MODEL_MAPPINGS.get(model_id, None)

        if model_cls is None:
            raise ValueError("load model from config failed")
        return model_cls
