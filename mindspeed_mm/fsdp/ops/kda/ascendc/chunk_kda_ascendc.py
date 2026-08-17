import torch
import warnings
from typing import Optional

from triton_ascend_kernels.attention.fla.kda.chunk import (
    ChunkKDAFunction,
    l2norm_fwd,
    fused_beta_sigmoid,
    prepare_chunk_indices,
    autocast_custom_fwd,
    input_guard,
)

from triton_ascend_kernels.attention.fla.kda.chunk_fwd import chunk_kda_fwd

def _host_int_tuple(value):
    """Convert tensor metadata to host int tuple (at most one D2H)."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(v) for v in value)


def _bnsd_to_bsnd(tensor):
    """Convert BNSD -> BSND for Triton backward compatibility."""
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(
            "AscendC KDA saved tensors must be rank-4 BNSD tensors, "
            f"but received shape {tuple(tensor.shape)}."
        )
    return tensor.permute(0, 2, 1, 3).contiguous()


_ascendc_chunk_kda_fwd = None
_ascendc_import_error = None


def _try_import_ascendc_kda_fwd():
    """Lazy import AscendC KDA forward, cache result."""
    global _ascendc_chunk_kda_fwd, _ascendc_import_error
    if _ascendc_chunk_kda_fwd is not None:
        return _ascendc_chunk_kda_fwd

    try:
        from fla_npu.ops.ascendc import chunk_kda_fwd as ascendc_chunk_kda_fwd
        _ascendc_chunk_kda_fwd = ascendc_chunk_kda_fwd
        return _ascendc_chunk_kda_fwd
    except Exception as e:
        _ascendc_import_error = e
        warnings.warn(
            "AscendC KDA forward not available, falling back to Triton. "
            f"Import error: {e}",
            RuntimeWarning,
        )
        return None


@staticmethod
@input_guard
@autocast_custom_fwd
def patched_forward(
    ctx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    transpose_state_layout: bool = False,
    skip_recompute: bool = False
):
    ascendc_chunk_kda_fwd = _try_import_ascendc_kda_fwd()

    if skip_recompute:
        if disable_recompute:
            raise ValueError("`skip_recompute` is not compatible with `disable_recompute=True`.")
        if return_intermediate_states:
            raise ValueError("`skip_recompute` is not compatible with `return_intermediate_states=True`.")
        # Lazy imports to avoid a hard dependency on mindspeed_mm (and circular imports).
        from mindspeed_mm.fsdp.train.training_context import TrainingContext, TrainingStage
        from mindspeed_mm.fsdp.features.memory.async_offload import OffloadManager, SwapTensor
        from mindspeed_mm.fsdp.utils.device import get_current_stream
        training_stage = TrainingContext().get_training_stage()
        layer_idx = TrainingContext().get_layer_index()
        depth = TrainingContext().get_model_depth()

    # Apply l2norm
    q_rstd, k_rstd = None, None
    if use_qk_l2norm_in_kernel:
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)

    beta_raw = beta
    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)

    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(
            cu_seqlens,
            chunk_size,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    g_input = g

    if ascendc_chunk_kda_fwd is None:
        # if ascendc is not available, use triton impl instead
        (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state) = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g_input,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
        )
    else:
        chunk_size = 64
        if skip_recompute and training_stage == TrainingStage.BACKWARD:
            # --- Recompute replay: restore the forward intermediates offloaded to host
            # instead of recomputing chunk_kda_fwd. Values are bit-identical to the
            # first forward, so gradients are identical to full recomputation. ---
            # Put order during forward: [o, Aqk, Akk, (final_state), (g_cumsum)].
            swap_tensor_nums = 3 + int(output_final_state) + int(not use_gate_in_kernel)
            if layer_idx == depth - 1:
                # Last layer's tensors were kept on device (LIFO stack).
                restored = [OffloadManager().pop_npu_tensor().tensor for _ in range(swap_tensor_nums)]
                restored.reverse()
            else:
                h2d_stream = OffloadManager().swap_stream
                restored = []
                for swap_key in reversed(OffloadManager().get_layer_items_keys(layer_idx)[-swap_tensor_nums:]):
                    swap_tensor = OffloadManager().get(swap_key)
                    swap_tensor.launch_h2d(h2d_stream)
                    get_current_stream().wait_event(swap_tensor.h2d_event)
                    restored.append(swap_tensor.tensor)
                    OffloadManager().clear(swap_key)
                restored.reverse()
            o, Aqk, Akk = restored[0], restored[1], restored[2]
            idx = 3
            final_state = None
            if output_final_state:
                final_state = restored[idx]
                idx += 1
            g_cumsum = None
            if not use_gate_in_kernel:
                g_cumsum = restored[idx]
            # Same as the default (disable_recompute=False) path: these are rebuilt
            # inside chunk_kda_bwd from the inputs and Aqk/Akk.
            w, u, qg, kg, v_new, h = None, None, None, None, None, None
        else:
            # The AscendC Python entry consumes Host metadata. Prefer the CPU
            # copy already prepared by the model to avoid another D2H.
            cu_seqlens_host = _host_int_tuple(
                cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
            )
            # The non-CP AscendC interface derives canonical sequence-major
            # chunk indices from cu_seqlens. Dense inputs carry no metadata.
            ascendc_chunk_indices = (
                None
                if cu_seqlens_host is not None
                else _host_int_tuple(chunk_indices)
            )

            (
                o,
                final_state,
                g_cumsum,
                Aqk,
                Akk,
                w,
                u,
                qg,
                kg,
                v_new,
                h,
                initial_state,
            ) = ascendc_chunk_kda_fwd(
                q,
                k,
                v,
                g_input,
                beta,
                float(scale),
                int(chunk_size),
                layout="BSND",
                initial_state=initial_state,
                output_final_state=bool(output_final_state),
                cu_seqlens=cu_seqlens_host,
                chunk_indices=ascendc_chunk_indices,
                safe_gate=bool(safe_gate),
                lower_bound=lower_bound,
                use_gate_in_kernel=bool(use_gate_in_kernel),
                A_log=A_log.float(),
                dt_bias=dt_bias.float(),
                disable_recompute=bool(disable_recompute),
                return_intermediate_states=bool(return_intermediate_states),
                state_v_first=bool(transpose_state_layout),
            )

            # AscendC keeps backward-facing intermediates in BNSD. The current
            # Triton backward consumes BSND, so every D2D conversion is explicit.
            g_cumsum = _bnsd_to_bsnd(g_cumsum)
            Aqk = _bnsd_to_bsnd(Aqk)
            Akk = _bnsd_to_bsnd(Akk)
            w = _bnsd_to_bsnd(w)
            u = _bnsd_to_bsnd(u)
            qg = _bnsd_to_bsnd(qg)
            kg = _bnsd_to_bsnd(kg)
            v_new = _bnsd_to_bsnd(v_new)

        if skip_recompute and training_stage == TrainingStage.FORWARD:
            # --- Forward pass: offload the intermediates needed by backward (and the
            # output o consumed by downstream ops during the replay) to host. ---
            swap_tensors = [o, Aqk, Akk]
            if output_final_state:
                swap_tensors.append(final_state)
            if not use_gate_in_kernel:
                swap_tensors.append(g_cumsum)
            d2h_stream = OffloadManager().swap_stream
            for swap_tensor in swap_tensors:
                key, after_block = OffloadManager().get_cnt(layer_idx)
                if after_block:
                    # Free device memory of the previous layer's tensors.
                    OffloadManager().del_npu_tensor("{}_".format(layer_idx - 1))
                if layer_idx == depth - 1:
                    # Keep the last layer's tensors on device; they are consumed first.
                    OffloadManager().put_npu_tensor(SwapTensor(swap_tensor, key))
                else:
                    swap_tensor = SwapTensor(swap_tensor, key)
                    swap_tensor.launch_d2h(d2h_stream)
                    OffloadManager().put(key, swap_tensor)

    if return_intermediate_states:
        assert torch.is_inference_mode_enabled(), "return_intermediate_states is only allowed in inference mode"
        assert disable_recompute is False, "return_intermediate_states must be used with disable_recompute=False"
        return o.type_as(q), final_state, h

    ctx.save_for_backward(
        q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta_raw, beta, A_log, dt_bias, Aqk, Akk,
        w, u, qg, kg, v_new, h,
        initial_state, cu_seqlens, chunk_indices
    )
    ctx.chunk_size = chunk_size
    ctx.safe_gate = safe_gate
    ctx.scale = scale
    ctx.lower_bound = lower_bound
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.use_gate_in_kernel = use_gate_in_kernel
    ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
    ctx.allow_neg_eigval = allow_neg_eigval
    ctx.disable_recompute = disable_recompute
    ctx.state_v_first = state_v_first
    return o.type_as(q), final_state

def apply_ascendc_chunk_kda_patch():
    ChunkKDAFunction.forward = patched_forward
