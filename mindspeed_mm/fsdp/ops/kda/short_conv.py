# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Short convolution implementation for efficient causal convolutions."""

import warnings

import torch
import torch.nn as nn
from einops import rearrange




class ShortConvolution(nn.Conv1d):
    """Short convolution layer for efficient causal convolution operations.

    This class implements a depthwise 1D convolution with causal padding,
    designed for efficient sequence processing. It supports multiple backends (Triton/CUDA)
    and optional activation functions.

    Args:
        hidden_size (int): Number of input/output channels (must be equal for depthwise conv)
        kernel_size (int): Size of the convolution kernel
        bias (bool, optional): Whether to include learnable bias. Defaults to False.
        activation (Optional[str], optional): Activation function ('silu' or 'swish'). Defaults to 'silu'.
        backend (Optional[str], optional): Backend implementation ('triton' or 'cuda') for the decode `step` path. Defaults to 'triton'.
        device (Optional[torch.device], optional): Device to place the layer on. Defaults to None.
        dtype (Optional[torch.dtype], optional): Data type for layer parameters. Defaults to None.
        implementation (str, optional): Implementation of the non-decode forward path,
            'triton' (default) or 'ascendc' (AscendC fused op from fla_npu, NPU only).
        head_num (Optional[int], optional): Number of attention heads packed into the channel dim,
            required by the AscendC causal_conv1d op. Defaults to None (treated as 1).
        **kwargs: Additional keyword arguments (deprecated 'use_fast_conv1d' supported for compatibility)

    Attributes:
        hidden_size (int): Number of channels
        activation (Optional[str]): Selected activation function
        backend (str): Actual backend being used (may differ from input due to availability)

    Note:
        - Uses depthwise convolution (groups=hidden_size) for efficiency
        - Applies causal padding (kernel_size-1) to ensure no future information leakage
        - Falls back to Triton backend if CUDA backend is unavailable
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        backend: str | None = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        implementation: str = 'triton',
        head_num: int | None = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        # Number of attention heads packed into the channel dim, required by the
        # AscendC causal_conv1d op. Defaults to 1 (whole channel dim as one head).
        self.head_num = head_num if head_num is not None else 1
        if implementation not in ('triton', 'ascendc'):
            raise ValueError(
                f"Unsupported causal conv1d implementation: {implementation}. "
                "Expected 'triton' or 'ascendc'."
            )
        self.implementation = implementation
        self.activation = None

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if 'use_fast_conv1d' in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead.",
            )

        self.backend = 'triton'

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        s += f', backend={self.backend}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        weight: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            chunk_indices (Optional[torch.LongTensor]):
                Chunk indices for variable-length sequences. Default: `None`.
            weight (`Optional[torch.Tensor]`):
                Optional weight override of shape `[D_local, 1, W]` (e.g. a head-sharded
                slice under Ulysses context parallel). Default: `None` (use `self.weight`).

        Returns:
            Tensor of shape `[B, T, D]`.
        """
        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # in decoding phase, the cache (if provided) is updated inplace
        if B * T == N:
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return y, cache

        if weight is None:
            weight = self.weight

        if self.implementation == 'ascendc':
            # Import here to avoid circular dependency and to keep torch_npu/fla_npu
            # optional for the triton path.
            from mindspeed_mm.fsdp.ops.gdn.causal_conv1d_ascendc import causal_conv1d_ascendc

            # The AscendC op needs the number of heads packed in the channel dim.
            # Derive it from the (possibly head-sharded, e.g. under Ulysses CP) weight
            # so a weight override scales H proportionally.
            channels_per_head = self.hidden_size // self.head_num
            if weight.shape[0] % channels_per_head != 0:
                raise ValueError(
                    f"weight channels ({weight.shape[0]}) must be a multiple of "
                    f"channels_per_head ({channels_per_head})"
                )
            H = weight.shape[0] // channels_per_head
            y, final_state = causal_conv1d_ascendc(
                x=x,
                weight=rearrange(weight, "d 1 w -> d w"),  # AscendC op expects [D, W] weight
                H=H,
                bias=self.bias,
                residual=residual,
                initial_state=cache,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
                output_final_state=output_final_state,
            )
            # The AscendC op returns the head layout [B, H, T, d]; convert back to [B, T, D]
            y = y.transpose(1, 2).reshape(B, T, -1)
            return y, final_state

        from mindspeed_ops.api.triton.convolution import causal_conv1d

        return causal_conv1d(
            x=x,
            weight=rearrange(weight, "d 1 w -> w d"),
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.modules.conv.triton.ops import causal_conv1d_update

        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        # Always initialise cache when None so the Triton kernel never
        # receives a None tensor. Return value still respects output_final_state
        # to maintain consistency with the non-step path in forward().
        if cache is None:
            cache = x.new_zeros(N, D, W)
        # NOTE: we follow the fast mode that updates the cache in-place
        if self.backend == 'triton':
            y, cache = causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
            return y, (cache if output_final_state else None)


    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size
