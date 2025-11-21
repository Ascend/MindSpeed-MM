import torch

from megatron.training import get_args
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from mindspeed_mm.models.common.module import MultiModalModule
from .projectors.deepseekvl_mlp import create_deepseekvl_mlp
from .projectors.internvl_mlp import InternVLMLP
from .projectors.multimodal_projector import MultimodalProjector
from .projectors.glm_mlp import GlmMLP
from .vision_encoders.clip_vit_model import CLIPViT
from .vision_encoders.internvit_model import InternViT
from .vision_encoders.qwen2vl_vit_model import Qwen2VLViT
from .vision_encoders.qwen3vl_vit_model import Qwen3VLViT
from .vision_encoders.siglip_vit_model import create_siglip_vit
from .vision_encoders.glm4v_vl_vit_model import GlmViT

VISION_ENCODER_MAPPINGS = {
    "clip": CLIPViT,
    "InternViT": InternViT,
    "qwen2vit": Qwen2VLViT,
    "qwen2_5_vit": Qwen2VLViT,
    "qwen3vit": Qwen3VLViT,
    "videoalign_vit": Qwen2VLViT,
    "SigLip": create_siglip_vit,
    "glm4v_vit": GlmViT,
}

VISION_PROJECTION_MAPPINGS = {
    "mlp": MultimodalProjector,
    "InternVLMLP": InternVLMLP,
    "lnmlp": MultimodalProjector,
    "DeepSeekVL2MLP": create_deepseekvl_mlp,
    "GlmMLP": GlmMLP,
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
            *args,
            **kwargs
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = config.vision_encoder is not None
        self.add_projector = config.vision_projector is not None and self.post_process
        self.projector = None  # 开pp时projector只在最后一张卡有projector，这里默认要设为None不然影响freeze
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

    def forward(self, images: torch.Tensor = None, image_grid_thw: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        if self.add_encoder:
            encoder_out = self.encoder(pixel_values=images, grid_thw=image_grid_thw, **kwargs)
        if isinstance(encoder_out, tuple):
            image_embeddings, window_index = encoder_out
        else:
            image_embeddings, window_index = encoder_out, None
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)
            if window_index is not None:
                reverse_indices = torch.argsort(window_index)
                image_embeddings = image_embeddings[reverse_indices, :]

        return image_embeddings


class Qwen2vlVisionModel(VisionModel):
    def forward(self, images: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(images, image_grid_thw)
        if isinstance(encoder_out, tuple):
            image_embeddings, window_index = encoder_out
        else:
            image_embeddings, window_index = encoder_out, None
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)
            if window_index is not None:
                reverse_indices = torch.argsort(window_index)
                image_embeddings = image_embeddings[reverse_indices, :]

        return image_embeddings


class Qwen3vlVisionModel(MultiModalModule):
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
            *args,
            **kwargs
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = config.vision_encoder is not None
        self.add_projector = config.vision_projector is not None and self.post_process
        self.projector = None  # projector is None except for last pp rank
        self.encoder = None
        if self.add_encoder:
            self.encoder = VISION_ENCODER_MAPPINGS[config.vision_encoder.model_id](
                config=config.vision_encoder,
                transformer_layer_spec=encoder_transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
                projector_config=config.vision_projector
            )
        if self.add_projector:
            self.projector = VISION_PROJECTION_MAPPINGS[config.vision_projector.model_id](
                config=config.vision_projector,
                submodules=projector_layer_spec,
            )
        
        self.config = config

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
            for name, param in module.named_parameters():
                if 'pos_embed' in name or 'deepstack_layer' in name:  # fix freeze behavior
                    continue
                param.requires_grad = False

    def forward(self, images: torch.Tensor, image_grid_thw: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        if self.add_encoder:
            if get_args().tensor_model_parallel_size > 1:
                all_patch_num = images.shape[0]
                
                tp_size = get_args().tensor_model_parallel_size
                spatial_merge_size = self.config.vision_encoder.spatial_merge_size
                temporal_patch_size = self.config.vision_encoder.temporal_patch_size
                tp_size = tp_size * spatial_merge_size * temporal_patch_size

                res_dim = all_patch_num % tp_size
                pad_size = 0
                if res_dim != 0:
                    pad_size = tp_size - res_dim  # patch to lcm of tp size and all_patch_num
                    zero_tensor = torch.zeros(pad_size, images.shape[1], dtype=images.dtype, device='npu')
                    images = torch.cat((images, zero_tensor), dim=0)
                    pad_thw = torch.tensor([[1, 2, pad_size // 2]], dtype=image_grid_thw.dtype, device='npu')  # s, t
                    image_grid_thw = torch.cat((image_grid_thw, pad_thw), dim=0)
            
            encoder_out = self.encoder(pixel_values=images, grid_thw=image_grid_thw)

        deepstack_feature_lists = []
        if len(encoder_out) == 3:
            image_embeddings, window_index, deepstack_feature_lists = encoder_out
        elif len(encoder_out) == 2:
            image_embeddings, window_index = encoder_out
        else:
            image_embeddings, window_index = encoder_out, None
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)
            image_embeddings = gather_from_sequence_parallel_region(image_embeddings, tensor_parallel_output_grad=False)
            if window_index is not None:
                reverse_indices = torch.argsort(window_index)
                image_embeddings = image_embeddings[reverse_indices, :]

        if get_args().tensor_model_parallel_size > 1:
            if res_dim != 0:
                image_embeddings = image_embeddings[: all_patch_num // 4]  # s*t
        return image_embeddings, deepstack_feature_lists
