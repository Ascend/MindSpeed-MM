# -*- coding: utf-8 -*-
"""
A small-operator (pure PyTorch) implementation of chunk_kda, functionally
equivalent to ``fla.ops.kda.chunk_kda`` (flash-linear-attention v0.5.2,
``triton/flash-linear-attention``), which is the reference implementation for
the call in modeling_kimi.py.

Usage (same call signature as in modeling_kimi.py):

    from .chunk_kda_naive import chunk_kda_naive

    o, recurrent_state = chunk_kda_naive(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=self.A_log, dt_bias=self.dt_bias,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=self.gate_lower_bound is not None,
        lower_bound=self.gate_lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,
    )

Semantics aligned with fla's chunk_kda (verified item by item):
1. l2norm: normalized in fp32 (eps=1e-6) and cast back to the input dtype,
   same as triton l2norm_fwd.
2. gate (use_gate_in_kernel=True), computed in fp32:
   - lower_bound is None: g = -exp(A_log) * softplus(g + dt_bias)
   - lower_bound given:   g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
   Same as triton kda_gate_chunk_cumsum.
3. cumsum: chunk-local (chunk_size=64) cumsum in fp32.
   The triton kernel works in log2 space (scale=RCP_LN2 + exp2), which is
   mathematically equivalent to exp; plain exp is used here.
4. beta: used as-is by default. With use_beta_sigmoid_in_kernel=True,
   beta = scale * sigmoid(beta_raw) in fp32 (scale = 2.0 if allow_neg_eigval
   else 1.0), same as fla's fused_beta_sigmoid.
   Note: the Ascend port ``triton_ascend_kernels.attention.fla.kda.chunk_kda``
   currently lacks this parameter (it is swallowed by **kwargs and ignored),
   so with use_beta_sigmoid_in_kernel=True this implementation follows the
   fla reference and intentionally differs from the current Ascend kernel
   until the port adds support.
5. scale: defaults to K ** -0.5, applied to q (inter-chunk part) and Aqk
   (intra-chunk part), same as triton chunk_gla_fwd_kernel_o /
   chunk_kda_fwd_kernel_intra_sub_chunk.
6. With transpose_state_layout=True, initial_state / final_state use the
   [N, H, V, K] layout (computation internally uses [N, H, K, V], with
   transposes at the boundaries).
7. cu_seqlens (varlen): the input is a flattened [1, total_T, ...] batch;
   each sequence is processed independently and initial_state / final_state
   are indexed by sequence along dim N.
8. safe_gate / disable_recompute are kernel performance hints with no
   semantic effect; they are accepted and ignored here.

Differences from the fused kernel:
- Backward propagation relies on torch autograd (the fused kernel uses
  custom backward kernels).
- All state computation runs in fp32. Bit-exact equality with the fused
  kernel is not expected; deviations are at the usual fused-kernel numerical
  error level.
"""

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Aligned with triton l2norm_fwd: normalize in fp32, cast back to the input dtype."""
    original_dtype = x.dtype
    x = x.float()
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """
    Gate transform aligned with the gate part of triton kda_gate_chunk_cumsum (fp32).

    Args:
        g: Raw gate input of shape [..., H, K].
        A_log: Shape [H].
        dt_bias: Shape [H * K], optional.
        lower_bound: Forget gate lower bound, optional.

    Returns:
        Log-space decay of shape [..., H, K], fp32.
    """
    H, K = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(H, K)
    A = A_log.float().view(H, 1).exp()
    if lower_bound is not None:
        g = lower_bound * torch.sigmoid(A * g)
    else:
        g = -A * F.softplus(g)
    return g


def _chunk_kda_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    return_h: bool = False,
):
    """
    Chunk KDA forward for equal-length sequences (small-op version,
    WY representation + inter-chunk recurrence).

    Args:
        q, k, g: [B, T, H, K], fp32 (q/k already l2-normalized, g already
            gate-transformed).
        v: [B, T, H, V], fp32.
        beta: [B, T, H], fp32, used as-is (no sigmoid).
        scale: Attention scale.
        initial_state: [B, H, K, V] or None.
        output_final_state: Whether to return the final state.
        chunk_size: Chunk size (64, same as the triton version).
        return_h: Whether to return the intermediate state h at each chunk entry.

    Returns:
        o: [B, T, H, V], fp32.
        final_state: [B, H, K, V] fp32 or None.
        h: [B, NT, H, K, V] fp32 or None.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]

    # switch to head-first layout [B, H, T, D]
    q, k, v, g, beta = [x.transpose(1, 2).contiguous() for x in (q, k, v, g, beta)]

    # pad to a multiple of chunk_size
    pad_size = (chunk_size - T % chunk_size) % chunk_size
    total_length = T + pad_size
    q = F.pad(q, (0, 0, 0, pad_size)) * scale
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    # reshape to [B, H, NT, C, D]
    q, k, v, g, k_beta, v_beta = [
        x.reshape(B, H, -1, chunk_size, x.shape[-1])
        for x in (q, k, v, g, k_beta, v_beta)
    ]
    NT = q.shape[2]

    # chunk-local cumsum, same as triton chunk_local_cumsum
    g = g.cumsum(dim=-2)

    # ---------- intra chunk: WY representation ----------
    # decay_mask[i, j] = exp(g_i - g_j)
    # Clamp the masked upper-triangular region to avoid exp overflow to inf
    # (masked to zero in the forward, but inf would produce NaN gradients in
    # the backward); 80 is below the fp32 exp overflow threshold (~88.7).
    mask_lower = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    g_diff = (g.unsqueeze(-2) - g.unsqueeze(-3)).clamp(max=80)
    decay_mask = g_diff.exp()

    # Akk[i, j] = -beta_i * <k_i, k_j> * exp(g_i - g_j), strictly lower triangular
    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay_mask).sum(dim=-1)
    attn = attn.masked_fill(mask_lower, 0)
    # forward substitution for (I + tril(Akk))^{-1} (row by row, same
    # recurrence as the triton kernel)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    u = attn @ v_beta                    # [B, H, NT, C, V]
    w = attn @ (k_beta * g.exp())        # [B, H, NT, C, K]

    # ---------- inter-chunk recurrence ----------
    if initial_state is None:
        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    else:
        S = initial_state.to(torch.float32)
    o = torch.zeros_like(v)
    h = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=q.device) if return_h else None

    mask_intra = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        if return_h:
            h[:, :, i] = S
        # inter: (q * exp(g)) @ S (q already carries the scale)
        attn_inter = (q_i * g_i.exp()) @ S
        # intra: Aqk[i, j] = <q_i, k_j> * exp(g_i - g_j), lower triangular incl. diagonal
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1)
        attn_intra = attn_intra.masked_fill(mask_intra, 0)
        # delta rule: remove the old value already stored in the state
        v_new = u_i - w_i @ S
        o[:, :, i] = attn_inter + attn_intra @ v_new
        # state update: decay the whole state by exp(g_last), then write this chunk's delta
        S = S * g_i[:, :, -1].exp().unsqueeze(-1) \
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new

    final_state = S if output_final_state else None

    o = o.reshape(B, H, total_length, V)[:, :, :T]
    o = o.transpose(1, 2).contiguous()
    return o, final_state, h


def chunk_kda_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    transpose_state_layout: bool = False,
    chunk_size: int = 64,
    **kwargs,
):
    r"""
    A small-operator (pure PyTorch) implementation of chunk_kda, with the
    same interface and semantics as ``fla.ops.kda.chunk_kda``
    (flash-linear-attention v0.5.2).

    Args:
        q (torch.Tensor): Queries of shape `[B, T, H, K]`.
        k (torch.Tensor): Keys of shape `[B, T, H, K]`.
        v (torch.Tensor): Values of shape `[B, T, H, V]`.
        g (torch.Tensor): Gate input of shape `[B, T, H, K]`.
            With `use_gate_in_kernel=True` this is the raw input and the gate
            transform is applied internally; otherwise it is expected to be
            the pre-computed log-space decay.
        beta (torch.Tensor): Betas of shape `[B, T, H]`.
            Raw values by default; with `use_beta_sigmoid_in_kernel=True`,
            `beta = scale * sigmoid(beta)` is applied internally in fp32
            (scale = 2.0 if `allow_neg_eigval` else 1.0).
        scale (Optional[float]): Attention scale, defaults to `K ** -0.5`.
        initial_state (Optional[torch.Tensor]): Initial state of shape
            `[N, H, K, V]` (`[N, H, V, K]` with `transpose_state_layout=True`),
            fp32. N equals B for equal-length inputs, or the number of
            sequences for varlen inputs.
        output_final_state (bool): Whether to return the final state.
        use_qk_l2norm_in_kernel (bool): Whether to apply l2norm to q/k internally.
        use_gate_in_kernel (bool): Whether to apply the KDA gate transform
            internally. When True, `A_log` ([H]) and the optional `dt_bias`
            ([H * K]) must be passed via kwargs.
        use_beta_sigmoid_in_kernel (bool): Whether to apply
            `scale * sigmoid(beta)` internally. Default: `False`.
        allow_neg_eigval (bool): Only takes effect together with
            `use_beta_sigmoid_in_kernel=True`, scaling sigmoid(beta) by 2.0.
            Default: `False`.
        cu_seqlens (Optional[torch.LongTensor]): Cumulative sequence lengths
            `[N+1]` for varlen mode; the input must be a flattened batch with
            B == 1.
        safe_gate (bool): Kernel performance hint, no semantic effect;
            accepted and ignored.
        lower_bound (Optional[float]): Forget gate lower bound, used together
            with `use_gate_in_kernel=True`.
        disable_recompute (bool): Kernel performance hint, no semantic
            effect; accepted and ignored.
        return_intermediate_states (bool): Whether to additionally return the
            intermediate state h at each chunk entry.
        transpose_state_layout (bool): Use the transposed state layout
            `[N, H, V, K]`.
        chunk_size (int): Chunk size, default 64 (same as the triton version).

    Returns:
        - `return_intermediate_states=False`: (o, final_state)
            o: `[B, T, H, V]`, same dtype as the input q.
            final_state: `[N, H, K, V]` (`[N, H, V, K]` with
            `transpose_state_layout=True`), fp32; None when
            `output_final_state=False`.
        - `return_intermediate_states=True`: (o, final_state, h)
            h: `[B, NT, H, K, V]` (`[B, NT, H, V, K]` with
            `transpose_state_layout=True`), same dtype as the input q.
            In varlen mode B == 1 and NT is the total number of chunks across
            all sequences.
    """
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), \
        "v must be of shape (batch size, seq len, num of head, head dim)."
    if cu_seqlens is not None:
        assert q.shape[0] == 1, \
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."

    if scale is None:
        scale = k.shape[-1] ** -0.5
    input_dtype = q.dtype

    # 1. l2norm (normalize in fp32, then cast back to the input dtype, same as triton l2norm_fwd)
    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    # 2. gate transform (fp32)
    if use_gate_in_kernel:
        g = kda_gate(g, A_log, dt_bias, lower_bound)
    else:
        g = g.float()
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()

    # 3. beta sigmoid in fp32 (same as fla's fused_beta_sigmoid)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)

    # 4. state layout: internally always [N, H, K, V]
    if transpose_state_layout and initial_state is not None:
        initial_state = initial_state.transpose(-1, -2)

    # 5. per-sequence computation (each sequence is independent in varlen mode)
    if cu_seqlens is None:
        o, final_state, h = _chunk_kda_core(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            chunk_size, return_h=return_intermediate_states,
        )
    else:
        cu = cu_seqlens.tolist()
        outs, states, hs = [], [], []
        for n in range(len(cu) - 1):
            s, e = cu[n], cu[n + 1]
            h0 = initial_state[n:n + 1] if initial_state is not None else None
            o_n, state_n, h_n = _chunk_kda_core(
                q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                scale, h0, output_final_state, chunk_size,
                return_h=return_intermediate_states,
            )
            outs.append(o_n)
            if output_final_state:
                states.append(state_n)
            if return_intermediate_states:
                hs.append(h_n)
        o = torch.cat(outs, dim=1)
        final_state = torch.cat(states, dim=0) if output_final_state else None
        h = torch.cat(hs, dim=1) if return_intermediate_states else None

    # 6. restore layout and dtype
    o = o.to(input_dtype)
    if transpose_state_layout:
        if final_state is not None:
            final_state = final_state.transpose(-1, -2)
        if h is not None:
            h = h.transpose(-1, -2)
    if h is not None:
        h = h.to(input_dtype)

    if return_intermediate_states:
        return o, final_state, h
    return o, final_state
