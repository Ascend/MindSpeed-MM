# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from mindspeed_mm.models.common.module import MultiModalModule
from .vision_encoders.clip_vit_model import CLIPViT
from .vision_encoders.internvit_model import InternViT
from .vision_encoders.qwen2vl_vit_model import Qwen2VLViT
from .projectors.multimodal_projector import MultimodalProjector
from .projectors.internvl_mlp import InternVLMLP


VISION_ENCODER_MAPPINGS = {
    "clip": CLIPViT,
    "InternViT": InternViT,
    "qwen2vit": Qwen2VLViT,
}

VISION_PROJECTION_MAPPINGS = {
    "mlp": MultimodalProjector,
    "InternVLMLP": InternVLMLP,
    "lnmlp": MultimodalProjector,
}


class VisionModel(MultiModalModule):
    """
    Instantiate a vision encoder model from config.

    Args:
        config (dict): the general config for Vision Model
        {
            "vision_encoder": {...},  # Config for the image encoder.
            "vision_projector": {...},  # Config for the image projector.
            "drop_vision_class_token": (bool),  # Drop vision class token(s) before input to the text decoder.
        }
    """
    def __init__(
        self,
        config: TransformerConfig,
        encoder_transformer_layer_spec: ModuleSpec = None,
        projector_layer_spec: ModuleSpec = None,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = config.vision_encoder is not None
        self.add_projector = config.vision_projector is not None and self.post_process
        self.projector = None # 开pp时projector只在最后一张卡有projector，这里默认要设为None不然影响freeze
        self.encoder = None
        if self.add_encoder:
            self.encoder = VISION_ENCODER_MAPPINGS[config.vision_encoder.model_id](
                config=config.vision_encoder,
                transformer_layer_spec=encoder_transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
        if self.add_projector:
            self.projector = VISION_PROJECTION_MAPPINGS[config.vision_projector.model_id](
                config=config.vision_projector,
                submodules=projector_layer_spec,
            )

    def set_input_tensor(self, input_tensor):
        self.encoder.set_input_tensor(input_tensor)

    def freeze(
        self,
        freeze_encoder: bool = False,
        freeze_projector: bool = False
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_encoder (bool): Freeze the image encoder module.
            freeze_projection (bool): Freeze the image projector module.
        """

        modules = []
        if freeze_encoder and self.encoder is not None:
            modules.append(self.encoder)
        if freeze_projector and self.projector is not None:
            modules.append(self.projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.add_encoder:
            image_embeddings = self.encoder(images)
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)
        return image_embeddings
    
    
class Qwen2vlVisionModel(VisionModel):
    def forward(self, images: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.encoder(images, image_grid_thw)
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)

        return image_embeddings 