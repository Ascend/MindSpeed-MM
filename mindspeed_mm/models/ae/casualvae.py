from typing import Tuple

from einops import rearrange
import numpy as np
import torch
from torch import nn

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.conv import Conv2d, CausalConv3d
from mindspeed_mm.models.common.attention import CausalConv3dAttnBlock
from mindspeed_mm.models.common.resnet_block import ResnetBlock2D, ResnetBlock3D
from mindspeed_mm.models.common.updownsample import (SpatialDownsample2x, TimeDownsample2x, SpatialUpsample2x, TimeUpsample2x, 
                                    TimeUpsampleRes2x, Downsample, Spatial2xTime2x3DDownsample, Spatial2xTime2x3DUpsample)
from mindspeed_mm.models.common.checkpoint import load_checkpoint


CASUALVAE_MODULE_MAPPINGS = {
    "Conv2d": Conv2d,
    "ResnetBlock2D": ResnetBlock2D,
    "CausalConv3d": CausalConv3d,
    "AttnBlock3D": CausalConv3dAttnBlock,
    "ResnetBlock3D": ResnetBlock3D,
    "Downsample": Downsample,
    "SpatialDownsample2x": SpatialDownsample2x,
    "TimeDownsample2x": TimeDownsample2x,
    "SpatialUpsample2x": SpatialUpsample2x,
    "TimeUpsample2x": TimeUpsample2x,
    "TimeUpsampleRes2x": TimeUpsampleRes2x,
    "Spatial2xTime2x3DDownsample": Spatial2xTime2x3DDownsample,
    "Spatial2xTime2x3DUpsample": Spatial2xTime2x3DUpsample
}


def model_name_to_cls(model_name):
    if model_name in CASUALVAE_MODULE_MAPPINGS:
        return CASUALVAE_MODULE_MAPPINGS[model_name]
    else:
        raise ValueError(f"Model name {model_name} not supported")


class CausalVAE(MultiModalModule):
    def __init__(
        self,
        from_pretrained: str = None,
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (),
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        q_conv: str = "CausalConv3d",
        encoder_conv_in: str = "CausalConv3d",
        encoder_conv_out: str = "CausalConv3d",
        encoder_attention: str = "AttnBlock3D",
        encoder_resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        encoder_spatial_downsample: Tuple[str] = (
                "SpatialDownsample2x",
                "SpatialDownsample2x",
                "SpatialDownsample2x",
                "",
        ),
        encoder_temporal_downsample: Tuple[str] = (
                "",
                "TimeDownsample2x",
                "TimeDownsample2x",
                "",
        ),
        encoder_mid_resnet: str = "ResnetBlock3D",
        decoder_conv_in: str = "CausalConv3d",
        decoder_conv_out: str = "CausalConv3d",
        decoder_attention: str = "AttnBlock3D",
        decoder_resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        decoder_spatial_upsample: Tuple[str] = (
                "",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
        ),
        decoder_temporal_upsample: Tuple[str] = ("", "", "TimeUpsample2x", "TimeUpsample2x"),
        decoder_mid_resnet: str = "ResnetBlock3D",
        tile_sample_min_size: int = 256,
        tile_sample_min_size_t: int = 33,
        tile_latent_min_size_t: int = 16,
        tile_overlap_factor: int = 0.125,
        vae_scale_factor: list = None,
        use_tiling: bool = False,
        use_quant_layer: bool = True,
        **kwargs
    ) -> None:
        super().__init__(config=None)
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_sample_min_size_t = tile_sample_min_size_t
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(hidden_size_mult) - 1)))

        self.tile_latent_min_size_t = tile_latent_min_size_t
        self.tile_overlap_factor = tile_overlap_factor
        self.vae_scale_factor = vae_scale_factor
        self.use_tiling = use_tiling
        self.use_quant_layer = use_quant_layer

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
        )
        if self.use_quant_layer:
            quant_conv_cls = model_name_to_cls(q_conv)
            self.quant_conv = quant_conv_cls(2 * z_channels, 2 * embed_dim, 1)
            self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)
        if from_pretrained is not None:
            load_checkpoint(self, from_pretrained)

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def encode(self, x):
        if self.use_tiling:
            if (x.shape[-1] > self.tile_sample_min_size
                or x.shape[-2] > self.tile_sample_min_size
                or x.shape[-3] > self.tile_sample_min_size_t):
                posterior = self.tiled_encode(x)
        else:
            h = self.encoder(x)
            if self.use_quant_layer:
                h = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(h)
        res = posterior.sample().mul_(0.18215)
        return res

    def decode(self, z, **kwargs):
        z = z / 0.18215
        if self.use_tiling:
            if (z.shape[-1] > self.tile_latent_min_size
                or z.shape[-2] > self.tile_latent_min_size
                or z.shape[-3] > self.tile_latent_min_size_t):
                dec = self.tiled_decode(z)
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec = self.decoder(z)
        dec = rearrange(dec, "b c t h w -> b t c h w").contiguous()
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + \
                               b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + \
                               b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_sample_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i + 1] + 1] for i in range(len(t_chunk_idx) - 1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        moments = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start: end]
            if idx != 0:
                moment = self.tiled_encode2d(chunk_x, return_moments=True)[:, :, 1:]
            else:
                moment = self.tiled_encode2d(chunk_x, return_moments=True)
            moments.append(moment)
        moments = torch.cat(moments, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def tiled_decode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_latent_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i + 1] + 1] for i in range(len(t_chunk_idx) - 1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        dec_ = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start: end]
            if idx != 0:
                dec = self.tiled_decode2d(chunk_x)[:, :, 1:]
            else:
                dec = self.tiled_decode2d(chunk_x)
            dec_.append(dec)
        dec_ = torch.cat(dec_, dim=2)
        return dec_

    def tiled_encode2d(self, x, return_moments=False):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[:, :, :,
                       i: i + self.tile_sample_min_size,
                       j: j + self.tile_sample_min_size,
                       ]
                tile = self.encoder(tile)
                if self.use_quant_layer:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)
        if return_moments:
            return moments
        return posterior

    def tiled_decode2d(self, z):

        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[:, :, :,
                       i: i + self.tile_latent_min_size,
                       j: j + self.tile_latent_min_size,
                       ]
                if self.use_quant_layer:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)


class Encoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: str = "Conv2d",
        conv_out: str = "CasualConv3d",
        attention: str = "AttnBlock",
        resnet_blocks: Tuple[str] = (
                "ResnetBlock2D",
                "ResnetBlock2D",
                "ResnetBlock2D",
                "ResnetBlock3D",
        ),
        spatial_downsample: Tuple[str] = (
                "Downsample",
                "Downsample",
                "Downsample",
                "",
        ),
        temporal_downsample: Tuple[str] = ("", "", "TimeDownsampleRes2x", ""),
        mid_resnet: str = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        if len(resnet_blocks) != len(hidden_size_mult):
            raise AssertionError(f"the length of resnet_blocks and hidden_size_mult must be equal")
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.nonlinearity = nn.SiLU()

        # ---- In ----
        self.conv_in = model_name_to_cls(conv_in)(
            3, hidden_size, kernel_size=3, stride=1, padding=1
        )

        # ---- Downsample ----
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    model_name_to_cls(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(model_name_to_cls(attention)(
                        in_channels=block_in,
                        out_channels=block_in
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if spatial_downsample[i_level]:
                down.downsample = model_name_to_cls(spatial_downsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res // 2
            if temporal_downsample[i_level]:
                down.time_downsample = model_name_to_cls(temporal_downsample[i_level])(
                    block_in, block_in
                )
            self.down.append(down)

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = model_name_to_cls(attention)(
            in_channels=block_in,
            out_channels=block_in
        )
        self.mid.block_2 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        # ---- Out ----
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = model_name_to_cls(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
            if hasattr(self.down[i_level], "time_downsample"):
                hs_down = self.down[i_level].time_downsample(hs[-1])
                hs.append(hs_down)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: str = "Conv2d",
        conv_out: str = "CasualConv3d",
        attention: str = "AttnBlock",
        resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        spatial_upsample: Tuple[str] = (
                "",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
        ),
        temporal_upsample: Tuple[str] = ("", "", "", "TimeUpsampleRes2x"),
        mid_resnet: str = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.nonlinearity = nn.SiLU()

        # ---- In ----
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = model_name_to_cls(conv_in)(
            z_channels, block_in, kernel_size=3, padding=1
        )

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = model_name_to_cls(attention)(
            in_channels=block_in,
            out_channels=block_in
        )
        self.mid.block_2 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # ---- Upsample ----
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    model_name_to_cls(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(model_name_to_cls(attention)(
                        in_channels=block_in,
                        out_channels=block_in
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if spatial_upsample[i_level]:
                up.upsample = model_name_to_cls(spatial_upsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res * 2
            if temporal_upsample[i_level]:
                up.time_upsample = model_name_to_cls(temporal_upsample[i_level])(
                    block_in, block_in
                )
            self.up.insert(0, up)

        # ---- Out ----
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = model_name_to_cls(conv_out)(
            block_in, 3, kernel_size=3, padding=1
        )

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
            if hasattr(self.up[i_level], "time_upsample"):
                h = self.up[i_level].time_upsample(h)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                       + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                       torch.pow(self.mean - other.mean, 2) / other.var
                       + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar
               + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
