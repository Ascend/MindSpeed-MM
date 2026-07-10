import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock, FluxAttention


from mindspeed_mm.tasks.rl.soragrpo.sora_grpo_model import SoraGRPOModel
from mindspeed_mm.tasks.rl.soragrpo.ops.npu_ops import NpuFusedRMSNorm

class FluxGRPOModel(SoraGRPOModel):
    def __init__(self, args, device):
        super().__init__()
        self.ae = self._init_ae(args, device)
        self.reward = self.initialize_reward_model(args, device)
        self.text_encoder = None
        # Diffuser model is large, load to host memory first then FSDP shard, so load it last uniformly
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


_original_flux_attention_init = FluxAttention.__init__


def _patched_flux_attention_init(self, *args, **kwargs):
    # origin __init__
    _original_flux_attention_init(self, *args, **kwargs)

    # ---- patch ----
    self.norm_q = NpuFusedRMSNorm(self.head_dim, eps=kwargs.get("eps", 1e-5))

    self.norm_k = NpuFusedRMSNorm(self.head_dim, eps=kwargs.get("eps", 1e-5))

    if hasattr(self, "norm_added_q"):
        self.norm_added_q = NpuFusedRMSNorm(self.head_dim, eps=kwargs.get("eps", 1e-5))

    if hasattr(self, "norm_added_k"):
        self.norm_added_k = NpuFusedRMSNorm(self.head_dim, eps=kwargs.get("eps", 1e-5))


FluxAttention.__init__ = _patched_flux_attention_init
