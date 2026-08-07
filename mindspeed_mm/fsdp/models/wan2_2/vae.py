import importlib
import warnings
from abc import abstractmethod

from einops import rearrange, repeat
import torch
import torch.nn as nn
from diffusers.utils.accelerate_utils import apply_forward_hook

from mindspeed_mm.fsdp.models.base_model import GenerativeBaseModel


class DiffusersAEModel(nn.Module):
    """Diffusers autoencoder wrapper used by the FSDP2 Wan2.2 path."""

    def __init__(self, model_name, config):
        super().__init__()
        module = importlib.import_module("diffusers")
        automodel = getattr(module, model_name)
        self.model_name = model_name
        self.model = automodel.from_pretrained(config["from_pretrained"], torch_dtype=config.pop("dtype", None))
        self.do_sample = config.get("do_sample", True)

        self._tiling = False
        use_tiling = config.pop("enable_tiling", False)
        self.tiling_param = None
        if use_tiling:
            self.enable_tiling(tiling_param=config.get("tiling_param", None))

        self.norm_latents = config.pop("norm_latents", True)
        self.norm_mode = config.pop("norm_mode", "value_shift_scale")

    def enable_tiling(self, tiling_param=None):
        has_custom_tiling_key = tiling_param is not None and (
            "tile_size" in tiling_param or "tile_stride" in tiling_param
        )
        if hasattr(self.model, "enable_tiling") and not has_custom_tiling_key:
            if tiling_param:
                self.model.enable_tiling(**tiling_param)
                self.tiling_param = tiling_param
            else:
                self.model.enable_tiling()
        else:
            self._tiling = True
            self.tiling_param = tiling_param if tiling_param else self.tiling_param
            warnings.warn(
                f"The autoencoder {self.model_name} in the diffusers library does not implement tiling functionality. "
                "Please ensure to call the custom tiling method to enable tiling. "
            )

    def disable_tiling(self):
        if hasattr(self.model, "disable_tiling"):
            self.model.disable_tiling()
        else:
            self._tiling = False

    @apply_forward_hook
    def encode(self, x, **kwargs):
        if self._tiling:
            output = self.tiled_encode(x, **kwargs)
        else:
            output = self.model.encode(x, return_dict=True, **kwargs)
            if self.do_sample:
                output = output.latent_dist.sample()
            else:
                output = output.latent_dist.mode()

        if self.norm_latents:
            output = self.normalize_latent(output)

        return output

    # Alias so the ModelContainer can call the VAE as a normal module while
    # still using the decorated encode path.
    forward = encode

    @abstractmethod
    def tiled_encode(self, x, **kwargs):
        pass

    def normalize_latent(self, x):
        if self.norm_mode == "value_shift_scale":
            if getattr(self.model.config, "shift_factor", None):
                output = (x - self.model.config.shift_factor) * self.model.config.scale_factor
            else:
                output = x * self.model.config.scale_factor
        elif self.norm_mode == "channel_specified_shift_scale":
            latents_mean = torch.tensor(self.model.config.latents_mean).view(1, -1, 1, 1, 1).to(x)
            latents_std = torch.tensor(self.model.config.latents_std).view(1, -1, 1, 1, 1).to(x)
            output = (x - latents_mean) / latents_std
        else:
            raise NotImplementedError(f"norm_mode: {self.norm_mode} is not implemented.")
        return output

    @apply_forward_hook
    def decode(self, x, **kwargs):
        if self._tiling:
            return self.tiled_decode(x, **kwargs)
        return self.model.decode(x).sample

    @abstractmethod
    def tiled_decode(self, x, **kwargs):
        pass


class WanVideoVAE(DiffusersAEModel, GenerativeBaseModel):
    """Wan2.2 VAE wrapper kept local to FSDP2 to avoid importing the Megatron AE package."""

    def __init__(self, **config):
        super().__init__(model_name="AutoencoderKLWan", config=config)
        self.upsampling_factor = 8

    @classmethod
    def from_pretrained(cls, config):
        return cls._from_config(config)

    @classmethod
    def _from_config(cls, config):
        cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        return cls(**cfg)

    def setup_weights(self, config):
        # Weights are loaded directly in ``__init__`` via diffusers
        # ``from_pretrained``; nothing to set up afterwards.  The explicit
        # no-op prevents ModelContainerHub from re-loading the checkpoint
        # through the GenerativeBaseModel default (whose keys would not match this
        # wrapper's ``model.``-prefixed state dict).
        pass

    def _build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def _build_mask(self, data, is_bound, border_width):
        _, _, _, h_size, w_size = data.shape
        h_mask = self._build_1d_mask(h_size, is_bound[0], is_bound[1], border_width[0])
        w_mask = self._build_1d_mask(w_size, is_bound[2], is_bound[3], border_width[1])

        h_mask = repeat(h_mask, "H -> H W", H=h_size, W=w_size)
        w_mask = repeat(w_mask, "W -> H W", H=h_size, W=w_size)

        mask = torch.stack([h_mask, w_mask]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_encode(self, x, **kwargs):
        _, _, t_size, h_size, w_size = x.shape
        size_h, size_w = self.tiling_param["tile_size"]
        stride_h, stride_w = self.tiling_param["tile_stride"]
        size_h, size_w = (
            size_h * self.upsampling_factor,
            size_w * self.upsampling_factor,
        )
        stride_h, stride_w = (
            stride_h * self.upsampling_factor,
            stride_w * self.upsampling_factor,
        )

        tasks = []
        for h_start in range(0, h_size, stride_h):
            if h_start - stride_h >= 0 and h_start - stride_h + size_h >= h_size:
                continue
            for w_start in range(0, w_size, stride_w):
                if w_start - stride_w >= 0 and w_start - stride_w + size_w >= w_size:
                    continue
                h_end, w_end = h_start + size_h, w_start + size_w
                tasks.append((h_start, h_end, w_start, w_end))

        out_t = (t_size + 3) // 4
        weight = torch.zeros(
            (1, 1, out_t, h_size // self.upsampling_factor, w_size // self.upsampling_factor)
        ).to(x)
        values = torch.zeros(
            (1, 16, out_t, h_size // self.upsampling_factor, w_size // self.upsampling_factor)
        ).to(x)

        for h_start, h_end, w_start, w_end in tasks:
            hidden_states_batch = x[:, :, :, h_start:h_end, w_start:w_end]
            hidden_states_batch = self.model.encode(hidden_states_batch).latent_dist
            hidden_states_batch = hidden_states_batch.sample() if self.do_sample else hidden_states_batch.mode()

            mask = self._build_mask(
                hidden_states_batch,
                is_bound=(h_start == 0, h_end >= h_size, w_start == 0, w_end >= w_size),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(x)

            target_h = h_start // self.upsampling_factor
            target_w = w_start // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def tiled_decode(self, x, **kwargs):
        _, _, t_size, h_size, w_size = x.shape
        size_h, size_w = self.tiling_param["tile_size"]
        stride_h, stride_w = self.tiling_param["tile_stride"]

        tasks = []
        for h_start in range(0, h_size, stride_h):
            if h_start - stride_h >= 0 and h_start - stride_h + size_h >= h_size:
                continue
            for w_start in range(0, w_size, stride_w):
                if w_start - stride_w >= 0 and w_start - stride_w + size_w >= w_size:
                    continue
                h_end, w_end = h_start + size_h, w_start + size_w
                tasks.append((h_start, h_end, w_start, w_end))

        out_t = t_size * 4 - 3
        weight = torch.zeros((1, 1, out_t, h_size * self.upsampling_factor, w_size * self.upsampling_factor)).to(x)
        values = torch.zeros((1, 3, out_t, h_size * self.upsampling_factor, w_size * self.upsampling_factor)).to(x)

        for h_start, h_end, w_start, w_end in tasks:
            hidden_states_batch = x[:, :, :, h_start:h_end, w_start:w_end]
            hidden_states_batch = self.model.decode(hidden_states_batch).sample

            mask = self._build_mask(
                hidden_states_batch,
                is_bound=(h_start == 0, h_end >= h_size, w_start == 0, w_end >= w_size),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(x)

            target_h = h_start * self.upsampling_factor
            target_w = w_start * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def get_tiling_state(self):
        return self._tiling
