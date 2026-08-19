"""Public API for CCE vocab-tile streaming chunk loss.

This module is Triton-free on purpose: the fused kernel + autograd Function live in
chunkloss_cce_kernels.py and are imported lazily, so environments without Triton can
still import and run the training pipeline as long as CCE loss is not enabled.

Algorithm origin: Apple Cut-Cross-Entropy (ml-cross-entropy), adapted for NPU.
Replaces the three-layer stack (ChunkLoss + calculate_lm_loss + fixed_cross_entropy) in chunkloss.py.
"""
import torch


def chunk_loss_cce_fused(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    shift_labels: torch.Tensor,
    vt: int = 4096,
    ignore_index: int = -100,
    seq_chunk_size: int = None,
) -> torch.Tensor:
    """CCE multi-stream chunk loss.

    Args:
        hidden_states: (B, S, H) or (N, H) bf16
        head_weight: (V, H) bf16
        shift_labels: (B, S) or (N,) int64
        vt: vocab tile size, recommended 4096.
            A (N, vt) tile is ~128MB(N=8192), much smaller than the full logits.
        ignore_index: default -100
        seq_chunk_size: optional outer sequence chunking along the flattened token
            dimension. When 0 < seq_chunk_size < N, the tokens are split into segments
            of seq_chunk_size and each segment runs CCE independently with its loss
            summed up. None/<=0/>=N keeps the single-segment behavior.

    Returns:
        scalar loss tensor

    Recommended vt choices:
        - vt=4096: default, balances tile count and per-tile kernel overhead
        - vt=8192: halves tile count, reduces host event overhead (but doubles slot memory)
    """
    from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss_cce_kernels import ChunkLossCceFused

    if hidden_states.ndim == 3:
        B, S, H = hidden_states.shape
        h = hidden_states.reshape(-1, H).contiguous()
        labels = shift_labels.reshape(-1).contiguous()
    else:
        h = hidden_states.contiguous()
        labels = shift_labels.contiguous()
    if seq_chunk_size and 0 < seq_chunk_size < h.shape[0]:
        loss = None
        for s in range(0, h.shape[0], seq_chunk_size):
            e = min(s + seq_chunk_size, h.shape[0])
            seg_loss = ChunkLossCceFused.apply(h[s:e], head_weight, labels[s:e], vt, ignore_index)
            loss = seg_loss if loss is None else loss + seg_loss
        return loss
    return ChunkLossCceFused.apply(h, head_weight, labels, vt, ignore_index)
