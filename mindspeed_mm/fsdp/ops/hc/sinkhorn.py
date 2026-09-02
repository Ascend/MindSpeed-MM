# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import triton
import torch
import torch.nn.functional as F

from mindspeed_mm.fsdp.ops.hc.triton.sinkhorn import (
    _hc_split_sinkhorn_kernel_part1,
    _hc_split_sinkhorn_kernel_part2,
    _hc_split_sinkhorn_backward_kernel_part1,
    _hc_split_sinkhorn_backward_kernel_part2,
)

__all__ = ["SinkhornFunction", "hc_split_sinkhorn"]


class SinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Triton implementation of HC-Split Sinkhorn.
        Args:
            mixes: input tensor, [batch_size, seq_len, (2+hc_mult)*hc_mult]
            hc_scale: scaling tensor, [3], corresponding to pre / post / comb in sequence
            hc_base: bias tensor, [(2+hc_mult)*hc_mult]
            hc_mult: HC dimension. Currently, only 4 is supported.
            sinkhorn_iters: number of Sinkhorn normalization iterations
            eps: small constant to prevent division by zero.

        Returns:
            (pre, post, comb)
                pre:  [batch_size, seq_len, hc_mult]
                post: [batch_size, seq_len, hc_mult]
                comb: [batch_size, seq_len, hc_mult, hc_mult]
        """
        if len(mixes.shape) != 3:
            raise ValueError("this op is not supported, when mixes.shape != 3")
        if hc_mult != 4:
            raise ValueError("hc_mult only support 4")

        # Flatten batch and sequence dimensions for Triton processing
        b, s, _ = mixes.shape
        feat_dim = (2 + hc_mult) * hc_mult
        batch_seq_size = b * s
        mixes_flat = mixes.view(-1, feat_dim).contiguous()

        # Initialize output tensors
        pre_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
        post_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
        comb_tmp = torch.empty((batch_seq_size, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device)

        # Configure Triton kernel parameters
        BLOCK_ALIGN = 8
        group_part1 = 64
        group_part2 = 32

        # Launch Part1 kernel (Pre/Post computation)
        _hc_split_sinkhorn_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
            mixes_flat,
            hc_scale,
            hc_base,
            pre_flat,
            post_flat,
            comb_tmp,
            batch_seq_size,
            eps,
            feat_dim,
            hc_mult,
            group_part1,
        )

        # Pad comb tensor for memory alignment
        comb_tmp_padded = F.pad(comb_tmp, pad=(0, BLOCK_ALIGN - hc_mult), mode="constant", value=float("-inf"))
        comb_flat_padded = torch.empty((batch_seq_size, hc_mult * BLOCK_ALIGN), dtype=mixes.dtype, device=mixes.device)

        # Launch Part2 kernel (Comb computation with Sinkhorn normalization)
        _hc_split_sinkhorn_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
            comb_tmp_padded,
            comb_flat_padded,
            batch_seq_size,
            hc_mult,
            sinkhorn_iters,
            eps,
            group_part2,
            BLOCK_ALIGN=BLOCK_ALIGN,
        )

        # Reshape outputs and restore original dtype
        pre = pre_flat.view(b, s, hc_mult)
        post = post_flat.view(b, s, hc_mult)
        comb = comb_flat_padded.view(b, s, hc_mult, BLOCK_ALIGN)[:, :, :, :hc_mult]

        ctx.save_for_backward(mixes, hc_scale, hc_base)

        ctx.hc_mult = hc_mult
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps

        return pre, post, comb

    @staticmethod
    def backward(
        ctx, grad_pre: torch.Tensor, grad_post: torch.Tensor, grad_comb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the gradients of mixes / hc_scale / hc_base.
        Args:
            grad_pre: upstream gradient of pre, with shape [b, s, hc_mult]
            grad_post: upstream gradient of post, with shape [b, s, hc_mult]
            grad_comb: upstream gradient of comb, with shape [b, s, hc_mult, hc_mult]

        Returns:
            (grad_mixes, grad_hc_scale, grad_hc_base, None, None, None)
        """
        mixes, hc_scale, hc_base = ctx.saved_tensors
        hc_mult = ctx.hc_mult
        sinkhorn_iters = ctx.sinkhorn_iters
        eps = ctx.eps

        # Input dimension validation
        b, s, _ = mixes.shape
        batch_seq_size = b * s

        # Initialize gradient tensors with zeros（两个 backward kernel 用 atomic_add 累加，必须清零）
        grad_mixes = torch.zeros_like(mixes, device=mixes.device)
        grad_hc_scale = torch.zeros_like(hc_scale, device=hc_scale.device)
        grad_hc_base = torch.zeros_like(hc_base, device=hc_base.device)
        comb_tmp = torch.empty((batch_seq_size, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device)

        # Flatten gradient tensors for Triton processing
        grad_pre_flat = grad_pre.reshape(-1, hc_mult)
        grad_post_flat = grad_post.reshape(-1, hc_mult)

        # Configure Triton kernel parameters
        BLOCK_ALIGN = 8
        group_part1 = 64
        group_part2 = 32

        # Launch Part1 kernel (Pre/Post gradients)
        _hc_split_sinkhorn_backward_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
            grad_pre_flat,
            grad_post_flat,
            mixes,
            hc_scale,
            hc_base,
            comb_tmp,
            grad_mixes,
            grad_hc_scale,
            grad_hc_base,
            batch_seq_size,
            hc_mult=hc_mult,
            group=group_part1,
        )

        # Prepare comb slice for Part2 backward kernel (padding for alignment)
        mixes_flat = mixes.view(-1, (2 + hc_mult) * hc_mult)
        mixes_slice = mixes_flat[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
        mixes_pad = F.pad(mixes_slice, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0.0)

        # Initialize padded gradient tensors
        grad_mixes_pad = torch.zeros(
            (batch_seq_size, hc_mult, BLOCK_ALIGN),
            dtype=grad_mixes.dtype,
            device=grad_mixes.device,
        )
        grad_hc_base_pad = torch.zeros((hc_mult, BLOCK_ALIGN), dtype=grad_hc_base.dtype, device=grad_hc_base.device)

        # Pad comb gradient tensor
        grad_comb_flat = grad_comb.reshape(-1, hc_mult, hc_mult)
        grad_comb_flat_pad = F.pad(grad_comb_flat, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0.0)
        comb_tmp_padded = F.pad(comb_tmp, pad=(0, BLOCK_ALIGN - hc_mult), mode="constant", value=float("-inf"))

        # If batch_seq_size is not an integer multiple of group_part2, the last block will read beyond the end of the tensor.
        # The number of rows of the three inputs is padded to an integer multiple, and the batch_seq_size passed to the kernel remains unchanged.
        row_pad = -batch_seq_size % group_part2
        if row_pad:
            mixes_pad = F.pad(mixes_pad, (0, 0, 0, 0, 0, row_pad))
            grad_comb_flat_pad = F.pad(grad_comb_flat_pad, (0, 0, 0, 0, 0, row_pad))
            comb_tmp_padded = F.pad(comb_tmp_padded, (0, 0, 0, 0, 0, row_pad))
        # Launch Part2 kernel (Comb gradients)
        _hc_split_sinkhorn_backward_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
            grad_comb_flat_pad,
            mixes_pad,
            hc_scale,
            comb_tmp_padded,
            grad_mixes_pad,
            grad_hc_scale,
            grad_hc_base_pad,
            batch_seq_size,
            hc_mult,
            sinkhorn_iters,
            eps,
            BLOCK_ALIGN=BLOCK_ALIGN,
            group=group_part2,
        )

        # Merge padded gradients back to original shape
        grad_mixes_slice = grad_mixes_pad[:, :, :hc_mult].reshape(b, s, hc_mult * hc_mult)
        grad_hc_base_slice = grad_hc_base_pad[:, :hc_mult].reshape(hc_mult * hc_mult)

        # Update final gradients
        grad_mixes[:, :, 2 * hc_mult :] = grad_mixes_slice
        grad_hc_base[2 * hc_mult :] = grad_hc_base_slice

        return grad_mixes, grad_hc_scale, grad_hc_base, None, None, None


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """External entry. The parameters and return value are the same as those of SinkhornFunction.forward.

    If the parameter is a DTensor, it is first converted into a local tensor. The Triton kernel uses raw pointers and cannot feed into DTensor.
    """
    from torch.distributed.tensor import DTensor

    def _local(x):
        return x.to_local() if isinstance(x, DTensor) else x

    return SinkhornFunction.apply(
        _local(mixes), _local(hc_scale), _local(hc_base), hc_mult, sinkhorn_iters, eps
    )
