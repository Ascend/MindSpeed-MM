import json

import torch
import transformers
from transformers.modeling_outputs import ModelOutput

from mindspeed_mm.models.text_encoder.hunyuan_mllm_text_encoder import HunyuanMLLmModel


class Hunyuan15MLLmModel(HunyuanMLLmModel):
    def __init__(
            self,
            model,
            template_info,
            image_embed_interleave=2,
    ):
        super().__init__(model=model, template_info=template_info, image_embed_interleave=image_embed_interleave)

    @classmethod
    def from_pretrained(cls, **config):
        template_file_path = config.pop("template_file_path")
        template_id = config.pop("template_id", "hyv-llm-encode-video")
        with open(template_file_path, "r") as f:
            templates = json.load(f)
        image_embed_interleave = config.pop("image_embed_interleave", 4)
        model_type = config.pop("model_type", "AutoModel")
        model = getattr(transformers, model_type).from_pretrained(**config)
        if hasattr(model, 'language_model'):
            model = model.language_model
        model.final_layer_norm = model.norm
        # from_pretrained will ensure that the model is in eval mode.
        model.requires_grad_(False)
        return Hunyuan15MLLmModel(
            model=model,
            template_info=templates[template_id],
            image_embed_interleave=image_embed_interleave,
        )
