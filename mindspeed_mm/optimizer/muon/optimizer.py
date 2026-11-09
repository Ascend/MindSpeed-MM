# Copyright (c) 2024 Keller Jordan
# Copyright (c) 2025 Moonshot AI
import math
import os
from typing import Tuple

import torch
from torch.distributed.tensor import DTensor

from .coefficients import resolve_ns_coefficients
from .distributed import (
    _muon_compute_placements,
    _per_expert_dtensor_update,
    _per_head_dtensor_update,
)
from .dtensor_utils import _from_local_like
from .newton_schulz import (
    newton_schulz_dtype,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_per_expert,
    zeropower_via_newtonschulz5_per_head,
)
from .owner import pipelined_owner_updates
from .owner_plan import collect_owner_batches
from .qk_clip import (
    _QK_CLIP_RUNTIME_KEYS,
    _apply_qk_clip_updates,
    _prepare_qk_clip_groups,
)


# Row-sharded 2-D weights are orthogonalized by an "owner" rank that gathers
# the full matrix (see owner.py). The exchange runs in "waves":
# - MINDSPEED_MM_MUON_OWNER_WAVE_BYTES: maximum bytes of full matrices that one
#   owner gathers in a wave (default 1 GiB). A single larger matrix gets a wave
#   of its own.
# - MINDSPEED_MM_MUON_OWNER_MAX_INFLIGHT: number of waves whose exchanges may be
#   in flight at the same time (default 2).
# - MINDSPEED_MM_MUON_OWNER_SCATTER_COMM: set to 1 to run the return exchange
#   on its own process group, so a gather need not wait for the previous return
#   exchange to retire off the shared stream (default 0, read in owner.py).
_OWNER_WAVE_BYTES = int(os.getenv("MINDSPEED_MM_MUON_OWNER_WAVE_BYTES", 1 << 30))
_OWNER_MAX_INFLIGHT = int(os.getenv("MINDSPEED_MM_MUON_OWNER_MAX_INFLIGHT", 2))


def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    """Return the Muon learning rate for one 2-D matrix of ``param_shape``.

    The multiplier ``sqrt(max(param_shape)) * matched_adamw_rms`` gives the
    orthogonalized update the per-element RMS of a typical AdamW update.
    Despite the name, only the learning rate is scaled; weight decay uses the
    base ``lr``.
    """
    if len(param_shape) != 2:
        raise ValueError(
            "adjust_lr_wd_for_muon expects the shape of one matrix, got "
            f"{tuple(param_shape)}"
        )
    A, B = param_shape
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def _prepare_muon_tensor(tensor, group):
    """Convert a Muon gradient or state tensor to its orthogonalization layout."""
    mode = group.get("muon_mode", "full_matrix")
    if mode == "full_matrix":
        if tensor.ndim > 2:
            return tensor.view(tensor.size(0), -1)
        return tensor
    if mode == "per_expert":
        if tensor.ndim != 3:
            raise ValueError(
                "per-expert Muon expects a 3-D parameter shaped "
                f"[num_experts, rows, cols], got {tuple(tensor.shape)}"
            )
        num_experts = group.get("muon_num_experts")
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError("per-expert Muon requires a positive integer muon_num_experts")
        if tensor.size(0) != num_experts:
            raise ValueError(
                f"parameter first dimension {tensor.size(0)} does not match "
                f"muon_num_experts={num_experts}"
            )
        return tensor
    if mode != "per_head":
        raise ValueError(f"unsupported muon_mode: {mode!r}")
    if tensor.ndim != 2:
        raise ValueError(
            "per-head Muon expects a 2-D parameter shaped "
            f"[num_heads * head_rows, head_cols], got {tuple(tensor.shape)}"
        )
    num_heads = group.get("muon_num_heads")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("per-head Muon requires a positive integer muon_num_heads")
    if tensor.size(0) % num_heads != 0:
        raise ValueError(
            f"parameter first dimension {tensor.size(0)} is not divisible by "
            f"muon_num_heads={num_heads}"
        )
    # Sharded tensors stay 2-D here; the per-head path splits them into heads
    # after gathering.
    if isinstance(tensor, DTensor):
        return tensor
    return tensor.view(num_heads, tensor.size(0) // num_heads, tensor.size(1))


def _muon_matrix_shape(tensor, group):
    """Return the logical matrix shape used for Muon's learning-rate scaling."""
    mode = group.get("muon_mode", "full_matrix")
    if mode == "per_head":
        if tensor.ndim == 2:
            return (tensor.size(0) // group["muon_num_heads"], tensor.size(1))
        return tensor.shape[1:]
    if mode == "per_expert":
        if tensor.ndim != 3:
            raise ValueError(
                "per-expert Muon expects [num_experts, rows, cols] for LR scaling, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.shape[1:]
    return tensor.shape


def _apply_muon_update(parameter, ns_input, group):
    """Orthogonalize one momentum tensor and apply it to its parameter."""
    mode = group.get("muon_mode", "full_matrix")
    is_dtensor = isinstance(parameter, DTensor)

    if is_dtensor and mode == "per_head":
        update = _per_head_dtensor_update(parameter, ns_input, group)
    elif is_dtensor and mode == "per_expert":
        update = _per_expert_dtensor_update(parameter, ns_input, group)
    else:
        update = None

    if update is None:
        # Fallback: all-gather the full matrices. Per-expert tensors keep their
        # expert-parallel shard, because experts are independent.
        if is_dtensor:
            placements = _muon_compute_placements(ns_input, mode)
            local = ns_input.redistribute(ns_input.device_mesh, placements).to_local()
            local_shape = local.shape
        else:
            local = ns_input

        # The update stays in the Newton-Schulz compute dtype (BF16 or FP16);
        # _apply_orthogonal_update adds it to the parameter with type promotion.
        if mode == "per_head":
            # Reshape the gathered 2-D tensor into complete heads.
            if local.ndim == 2:
                local = _prepare_muon_tensor(local, group)
            update = zeropower_via_newtonschulz5_per_head(
                local,
                num_heads=group["muon_num_heads"],
                steps=group["ns_steps"],
                coefficients=group.get("ns_coefficients"),
            )
        elif mode == "per_expert":
            update = local.to(
                newton_schulz_dtype(local.size(1), local.size(2), group["ns_steps"])
            ) if local.size(0) == 0 else zeropower_via_newtonschulz5_per_expert(
                # EP may make this a local subset of experts.
                local,
                num_experts=local.size(0),
                steps=group["ns_steps"],
                coefficients=group.get("ns_coefficients"),
            )
        else:
            update = zeropower_via_newtonschulz5(
                local,
                steps=group["ns_steps"],
                coefficients=group.get("ns_coefficients"),
            )

        if is_dtensor:
            update = _from_local_like(
                update.reshape(local_shape), parameter, placements
            ).redistribute(parameter.device_mesh, parameter.placements)

    _apply_orthogonal_update(parameter, ns_input, update, group)


def _apply_orthogonal_update(parameter, ns_input, update, group):
    """Decay the parameter and apply one orthogonalized update to it."""
    lr = group["lr"]
    weight_decay = group["weight_decay"]
    adjusted_lr = adjust_lr_wd_for_muon(
        lr, group["matched_adamw_rms"], _muon_matrix_shape(ns_input, group)
    )
    parameter.mul_(1 - lr * weight_decay)
    if adjusted_lr != 0:
        # The iteration may return a non-contiguous transposed matrix. The
        # update is in the Newton-Schulz compute dtype (BF16 or FP16); add_
        # promotes it and computes in the parameter's dtype.
        parameter.add_(update.reshape_as(parameter), alpha=-adjusted_lr)


def _migrate_legacy_momentum_buffers(state_dict, fallback_momentum):
    """Convert legacy ``muon_buffer`` (running sum) to ``momentum_buffer`` (EMA).

    Scales each legacy buffer by ``(1 - momentum)`` to match the EMA recurrence.
    Uses the momentum recorded in the checkpoint; falls back to
    ``fallback_momentum``. Does not mutate the incoming dict.
    """
    state = state_dict.get("state")
    if not isinstance(state, dict) or not any(
        isinstance(entry, dict) and "muon_buffer" in entry for entry in state.values()
    ):
        return state_dict

    # Use the momentum from the checkpoint (falls back to fallback_momentum).
    momentum_by_index = {}
    for group in state_dict.get("param_groups", ()):
        if not isinstance(group, dict):
            continue
        for index in group.get("params", ()):
            momentum_by_index[index] = group.get("momentum")

    migrated_state = dict(state)
    for index, entry in state.items():
        if not isinstance(entry, dict) or "muon_buffer" not in entry:
            continue
        entry = dict(entry)
        legacy = entry.pop("muon_buffer")
        # Both keys present: keep momentum_buffer, drop the stale one.
        if "momentum_buffer" not in entry:
            momentum = momentum_by_index.get(index)
            if momentum is None:
                momentum = fallback_momentum
            entry["momentum_buffer"] = legacy * (1 - momentum)
        migrated_state[index] = entry

    migrated = dict(state_dict)
    migrated["state"] = migrated_state
    return migrated


class Muon(torch.optim.Optimizer):
    """Muon (MomentUm Orthogonalized by Newton-Schulz) with an AdamW fallback.

    Groups marked ``use_muon`` use SGD momentum followed by Newton-Schulz
    orthogonalization of the update. Other groups use AdamW.

    Per-group keys set by ``muon_routing._mark_muon_param_groups``:
        use_muon: Whether the group uses Muon (otherwise the AdamW fallback).
        muon_mode: "full_matrix" (default), "per_head" or "per_expert".
        muon_num_heads / muon_num_experts: Head or expert count for the
            per-head and per-expert modes.
        param_names: Parameter names, used by the cross-rank consistency check.
        qk_clip_*: QK-Clip settings, see ``qk_clip.py``.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: Base learning rate shared by Muon and AdamW groups.
        weight_decay: Decoupled weight decay.
        matched_adamw_rms: Muon updates are scaled to match the per-element
            RMS of a typical AdamW update (0.2 to 0.4 recommended).
        momentum: Momentum for Muon groups.
        nesterov: Whether Muon groups use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        ns_coefficients: A named or explicit Newton-Schulz coefficient schedule.
        qk_clip_threshold: QK-Clip threshold; ``None`` disables clipping.
        qk_clip_alpha: Exponent for query scaling; key uses 1 - alpha.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(self,
                 param_groups,
                 lr: float = 2e-2,
                 weight_decay: float = 0.1,
                 matched_adamw_rms: float = 0.2,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 ns_steps: int = 5,
                 ns_coefficients=None,
                 adamw_betas: Tuple[float, float] = (0.95, 0.95),
                 adamw_eps: float = 1e-8,
                 qk_clip_threshold=None,
                 qk_clip_alpha: float = 0.5):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=resolve_ns_coefficients(ns_coefficients),
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            qk_clip_threshold=qk_clip_threshold,
            qk_clip_alpha=qk_clip_alpha,
        )

        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            group["ns_coefficients"] = resolve_ns_coefficients(
                group.get("ns_coefficients")
            )

    def state_dict(self):
        """Save optimizer state, omitting QK-Clip tensors, providers and process groups."""
        state_dict = super().state_dict()
        for group in state_dict["param_groups"]:
            for key in _QK_CLIP_RUNTIME_KEYS:
                group.pop(key, None)
        return state_dict

    def load_state_dict(self, state_dict):
        """Restore optimizer state and reattach the current QK-Clip runtime bindings."""
        runtime_metadata = [
            {key: group[key] for key in _QK_CLIP_RUNTIME_KEYS if key in group}
            for group in self.param_groups
        ]

        state_dict = _migrate_legacy_momentum_buffers(
            state_dict, self.defaults["momentum"]
        )
        result = super().load_state_dict(state_dict)
        for group, metadata in zip(self.param_groups, runtime_metadata):
            group["ns_coefficients"] = resolve_ns_coefficients(
                group.get("ns_coefficients")
            )
            for key in _QK_CLIP_RUNTIME_KEYS:
                group.pop(key, None)
            group.update(metadata)
        return result

    def _muon_ns_input(self, parameter, group):
        """Advance the momentum buffer and return the tensor to orthogonalize."""
        state = self.state[parameter]
        grad = _prepare_muon_tensor(parameter.grad, group)
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - group["momentum"])
        return grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf

    def _owner_compute_matrices(self, entries, process_group):
        """Orthogonalize one batch of row-sharded matrices with the owner exchange (owner.py)."""
        shapes = [(rows, columns) for _, _, rows, columns in entries]
        steps = entries[0][1]["ns_steps"]
        coefficients = entries[0][1].get("ns_coefficients")
        ns_inputs = [
            self._muon_ns_input(parameter, group)
            for parameter, group, _, _ in entries
        ]
        local_updates = pipelined_owner_updates(
            [ns_input.to_local() for ns_input in ns_inputs],
            shapes,
            process_group,
            steps,
            wave_bytes=_OWNER_WAVE_BYTES,
            coefficients=coefficients,
            max_inflight=_OWNER_MAX_INFLIGHT,
        )
        for (parameter, group, _, _), ns_input, local_update in zip(
            entries, ns_inputs, local_updates
        ):
            update = _from_local_like(local_update, parameter)
            _apply_orthogonal_update(parameter, ns_input, update, group)

    def _collect_owner_matrices(self):
        """Check cross-rank agreement and batch the row-sharded weights (owner_plan.py)."""
        return collect_owner_batches(self.param_groups, wave_bytes=_OWNER_WAVE_BYTES,
                                     max_inflight=_OWNER_MAX_INFLIGHT)

    def _step_muon(self):
        """Apply the Muon update to every parameter in a ``use_muon`` group."""
        deferred, deferred_ids = self._collect_owner_matrices()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            for p in group["params"]:
                if p.grad is None or id(p) in deferred_ids:
                    continue
                ns_input = self._muon_ns_input(p, group)
                _apply_muon_update(p, ns_input, group)

        for process_group, entries in deferred.values():
            self._owner_compute_matrices(entries, process_group)

    def _step_adamw(self):
        """Apply AdamW to every parameter outside a ``use_muon`` group."""
        for group in self.param_groups:

            if group.get('use_muon', False):
                continue

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            step = group['step']
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']

            params, grads, exp_avgs, exp_avg_sqs = [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                for key in ('adamw_exp_avg', 'adamw_exp_avg_sq'):
                    if key not in state:
                        state[key] = torch.zeros_like(p.grad)

                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['adamw_exp_avg'])
                exp_avg_sqs.append(state['adamw_exp_avg_sq'])

            if not params:
                continue

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
            squared = torch._foreach_mul(grads, grads)
            torch._foreach_lerp_(exp_avg_sqs, squared, 1 - beta2)

            denominator = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denominator, bias_correction2**0.5)
            torch._foreach_add_(denominator, eps)
            update = torch._foreach_div(exp_avgs, denominator)

            torch._foreach_mul_(params, 1 - lr * weight_decay)
            torch._foreach_add_(params, update, alpha=-lr / bias_correction1)

    def _preflight_qk_clip(self):
        """Consume, reduce and validate QK-Clip metrics. Returns clip factors."""
        return _prepare_qk_clip_groups(self.param_groups)

    def _apply_qk_clip(self, updates):
        """Clip query and key projections using already validated factors."""
        _apply_qk_clip_updates(updates)

    @torch.no_grad()
    def step(self, closure=None):
        """Run one optimizer step.

        ``closure`` follows the ``torch.optim.Optimizer`` interface. PyTorch
        distributed checkpointing calls ``step(closure=None)`` with zero
        gradients and a zero learning rate to create the optimizer state before
        loading or saving it; that step leaves the weights unchanged.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate QK-Clip metrics before any parameter update.
        qk_clip_updates = self._preflight_qk_clip()
        self._step_muon()
        self._step_adamw()
        self._apply_qk_clip(qk_clip_updates)
        return loss
