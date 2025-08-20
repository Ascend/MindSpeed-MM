import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock

from mindspeed_mm.tasks.rl.soragrpo.sora_grpo_model import SoraGRPOModel


class FluxGRPOModel(SoraGRPOModel):
    def __init__(self, args, device):
        super().__init__()
        self.ae = self._init_ae(args, device)
        self.reward = self.initialize_reward_model(args, device)
        self.text_encoder = None
        # diffuser模型较大，且需要加载入host内存后续再使用FSDP分片，所以统一最后加载
        self.diffuser = self._init_diffuser(args)

    def _init_diffuser(self, args):
        return FluxTransformer2DModel.from_pretrained(
            args.load,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )

    def _init_ae(self, args, device):
        return AutoencoderKL.from_pretrained(
            args.load,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device)

    def get_split_modules(self):
        return FluxTransformerBlock, FluxSingleTransformerBlock