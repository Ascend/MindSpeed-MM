# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig

from mindspeed_mm.fsdp.checkpoint.hf_utils import read_hf_tensor_shape, read_tensor
from mindspeed_mm.fsdp.utils.constants import (
    DCP_DOWN_PROJ_SUFFIX,
    DCP_GATE_UP_PROJ_SUFFIX,
    HF_GATE_PROJ_SUFFIX,
    HF_UP_PROJ_SUFFIX,
)


def permute_moe_expert(
    key: str,
    tensor: torch.Tensor,
    patterns: List[str],
) -> torch.Tensor:
    """Permute MoE expert weights.

    - gate_up_proj: (num_experts, 2 * intermediate, hidden) <-> (num_experts, hidden, 2 * intermediate)
    - down_proj:    (num_experts, hidden, intermediate) <-> (num_experts, intermediate, hidden)

    Returns the tensor unchanged if the key does not match any pattern.
    """
    for pattern in patterns:
        if re.fullmatch(pattern, key):
            return tensor.permute(0, 2, 1).contiguous()
    return tensor


def reshape_fused_linear(
    key: str,
    tensor: torch.Tensor,
    fused_linear_names: Optional[List[str]],
) -> torch.Tensor:
    """Reshape fused-linear weights.

    (expert_num, input_dim, output_dim) -> (expert_num * input_dim, output_dim)

    No-op when ``fused_linear_names`` is None or empty.
    """
    if not fused_linear_names:
        return tensor
    if any(name in key for name in fused_linear_names):
        return tensor.view(-1, tensor.shape[-1])
    return tensor


def rename_key(key: str, hf_prefix: str, dcp_prefix: str) -> str:
    """Apply the (hf_prefix -> dcp_prefix) rename.
    """
    if len(hf_prefix) > 0:
        return key.replace(hf_prefix, dcp_prefix, 1)
    return f"{dcp_prefix}{key}"


def convert_diffusers_key(
    key: str,
    convert_mapping: Dict[str, str],
    str_replace_mapping: Dict[str, str],
) -> Optional[str]:
    """Convert one diffusers key to a native key using the supplied exact
    ``convert_mapping`` and ordered ``str_replace_mapping``.

    Returns ``None`` for keys to skip (e.g. ``_extra_state``).
    """
    # Skip diffusers internal keys
    if key.endswith("_extra_state"):
        return None

    # Exact key rename
    if key in convert_mapping:
        return convert_mapping[key]

    # Sub-string replacements (applied in order; single-character patterns are
    # skipped deliberately to avoid destructive replaces)
    for old_str, new_str in str_replace_mapping.items():
        if len(old_str) > 1:
            key = key.replace(old_str, new_str)

    return key


def merge_moe_expert_weights(
    state_dict: Dict[str, torch.Tensor],
    hf_keys: Tuple[str, ...],
    dcp_key: str,
    dcp_down_proj_suffix: str = DCP_DOWN_PROJ_SUFFIX,
    dcp_gate_up_proj_suffix: str = DCP_GATE_UP_PROJ_SUFFIX,
    hf_gate_proj_suffix: str = HF_GATE_PROJ_SUFFIX,
    hf_up_proj_suffix: str = HF_UP_PROJ_SUFFIX,
    transpose: bool = True,
    *,
    out: Optional[torch.Tensor] = None,
) -> Optional[Tuple[str, torch.Tensor]]:
    """Merge experts with the original concat/stack behavior or stream into ``out``.

    Missing keys return None without consuming inputs. The out path requires
    exact shape/dtype/device and releases each source before reading the next.
    A failed copy may leave out partially written; callers must not publish it.
    """
    if not all(key in state_dict for key in hf_keys):
        return None
    if dcp_key.endswith(dcp_down_proj_suffix):
        gate_up = False
        groups = [(key,) for key in hf_keys]
    elif dcp_key.endswith(dcp_gate_up_proj_suffix):
        gate_up = True
        groups = list(zip(hf_keys[::2], hf_keys[1::2]))
    else:
        raise ValueError(f'Unsupported DCP expert weight key: {dcp_key}')

    if out is not None:
        if not isinstance(out, torch.Tensor) or out.ndim != 3 or out.layout != torch.strided:
            raise ValueError('out must be a dense 3D expert tensor')
        if not hf_keys or len(set(hf_keys)) != len(hf_keys):
            raise ValueError('Expert keys must be nonempty and unique')
        if out.shape[0] != len(groups) or (gate_up and len(hf_keys) % 2):
            raise ValueError('Expert keys must form complete groups matching out expert count')
        axis = 1 if transpose else 0
        if gate_up and out.shape[axis + 1] % 2:
            raise ValueError('out fused gate/up dimension must be even')
        width = out.shape[axis + 1] // 2

    if gate_up:
        for gate_key, up_key in groups:
            if (gate_key.removesuffix(hf_gate_proj_suffix) != up_key.removesuffix(hf_up_proj_suffix)
                    or (out is not None and (not gate_key.endswith(hf_gate_proj_suffix)
                                             or not up_key.endswith(hf_up_proj_suffix)))):
                raise ValueError(f'Mismatched gate/up expert keys: {gate_key}, {up_key}')

    merged = []
    for expert, keys in enumerate(groups):
        if out is None:
            # Preserve dtype promotion, autograd and unequal gate/up row counts.
            weight = (torch.concat([state_dict.pop(key) for key in keys]) if gate_up
                      else state_dict.pop(keys[0]))
            merged.append(weight.T if transpose else weight)
        else:
            for projection, key in enumerate(keys):
                target = out[expert].narrow(axis, projection * width, width) if gate_up else out[expert]
                value = state_dict.pop(key)
                if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.layout != torch.strided:
                    raise ValueError(f'Expected a dense 2D expert matrix: {key}')
                value = value.T if transpose else value
                if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
                    raise ValueError(f'Expert out shape/dtype/device mismatch: {key}')
                with torch.no_grad():
                    target.copy_(value)
                del value  # Release the source (including a transpose view) before the next pop.
    return dcp_key, torch.stack(merged) if out is None else out


def split_moe_expert_weights(
    tensor: torch.Tensor,
    hf_keys: Tuple[str, ...],
    dcp_key: str,
    dcp_down_proj_suffix: str = DCP_DOWN_PROJ_SUFFIX,
    dcp_gate_up_proj_suffix: str = DCP_GATE_UP_PROJ_SUFFIX,
    hf_gate_proj_suffix: str = HF_GATE_PROJ_SUFFIX,
    hf_up_proj_suffix: str = HF_UP_PROJ_SUFFIX,
    transpose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Split a merged MTP expert tensor into individual HF tensors.
    """
    result = {}
    if dcp_key.endswith(dcp_gate_up_proj_suffix):
        # hf_keys preserves the (gate, up) order created during mapping, so even/odd slices align.
        for expert_tensor, gate_key, up_key in zip(
            tensor.unbind(), hf_keys[::2], hf_keys[1::2]
        ):
            gate_prefix = gate_key.removesuffix(hf_gate_proj_suffix)
            up_prefix = up_key.removesuffix(hf_up_proj_suffix)
            if gate_prefix != up_prefix:
                raise ValueError(f"Mismatched gate/up expert keys: {gate_key}, {up_key}")
            if transpose:
                expert_tensor = expert_tensor.T
            gate_proj, up_proj = torch.chunk(expert_tensor, 2, dim=0)
            result[gate_key] = gate_proj.contiguous()
            result[up_key] = up_proj.contiguous()
    elif dcp_key.endswith(dcp_down_proj_suffix):
        for expert_tensor, hf_key in zip(tensor.unbind(), hf_keys):
            if transpose:
                expert_tensor = expert_tensor.T
            result[hf_key] = expert_tensor.contiguous()
    else:
        raise ValueError(f"Unsupported DCP expert weight key: {dcp_key}")
    return result


def build_weight_transform(model_id: str, hf_dir: str) -> Optional["WeightTransformPipeline"]:
    transform_cls = WEIGHT_TRANSFORM_PIPELINES.get(model_id)
    return transform_cls(hf_dir=hf_dir) if transform_cls is not None else None


class WeightTransformPipeline(ABC):
    """Base pipeline for weight format conversion.
    """

    def __init__(self) -> None:
        self.hf_to_dcp_mapping: Dict[Tuple[str, ...], str] = {}
        self.dcp_to_hf_mapping: Dict[str, Tuple[str, ...]] = {}

    @staticmethod
    def _to_full(tensor: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor
        return tensor.full_tensor().cpu() if isinstance(tensor, DTensor) else tensor

    @abstractmethod
    def hf_to_dcp(
        self,
        key: str,
        tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Convert one HuggingFace tensor into a DCP tensor.

        Returns ``None`` when the input tensor has been consumed or buffered but
        does not produce an immediate DCP entry. For example, a pipeline may
        buffer several expert tensors and emit their merged tensor only after
        the complete group has arrived.
        """
        pass

    @abstractmethod
    def dcp_to_hf(
        self,
        key: str,
        tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert one DCP tensor into one or more HuggingFace tensors.

        Multiple tensors are returned when a DCP tensor stores weights that
        were merged during HF-to-DCP conversion. For example, a merged MoE
        expert tensor is split back into the individual ``gate_proj``,
        ``up_proj`` and/or ``down_proj`` HF expert weights.
        """
        pass


class KimiK31WeightTransformPipeline(WeightTransformPipeline):
    """Kimi-K3.1 weight mappings and padding rules.

    Group descriptors and preallocated expert merging support low-memory loads.
    """

    _DTYPES = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}
    _EXPERT_KWARGS = dict(hf_gate_proj_suffix='.w1.weight', hf_up_proj_suffix='.w3.weight', transpose=True)
    _EXPERT_RE = re.compile(
        r'^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$')

    def __init__(self, hf_dir: str, mtp_num_layers: Optional[int] = None) -> None:
        super().__init__()
        self.hf_dir = Path(hf_dir).resolve()
        with (self.hf_dir / 'config.json').open(encoding='utf-8') as stream:
            self.config = json.load(stream)
        text = self.config['text_config']
        self.num_layers = text['num_hidden_layers']
        self.num_experts = text['num_experts']
        self.hidden_size = text['hidden_size']
        expert_hidden_size = text.get('routed_expert_hidden_size')
        self.expert_hidden_size = self.hidden_size if expert_hidden_size is None else expert_hidden_size
        self.intermediate_size = text['moe_intermediate_size']
        dense = text['first_k_dense_replace']
        frequency = text.get('moe_layer_freq', 1)
        self.moe_layers = tuple(i for i in range(dense, self.num_layers) if i % frequency == 0)
        self.linear_heads = (text.get('linear_attn_config') or {}).get('num_heads')
        # mtp_num_layers is accepted for the common pipeline interface; K3.1 has no MTP conversion.
        for layer in self.moe_layers:
            prefix = f'language_model.model.layers.{layer}.block_sparse_moe.experts'
            gate = tuple(f'{prefix}.{e}.{part}.weight' for e in range(self.num_experts) for part in ('w1', 'w3'))
            down = tuple(f'{prefix}.{e}.w2.weight' for e in range(self.num_experts))
            self.hf_to_dcp_mapping[gate] = prefix + '.gate_up_proj'
            self.hf_to_dcp_mapping[down] = prefix + '.down_proj'
        self.dcp_to_hf_mapping = {v: k for k, v in self.hf_to_dcp_mapping.items()}
        self._buffered_weights = {}
        self._finished_targets = set()
        self._expert_dtypes = {}
        self._source_shapes = {}
        self._source_metadata = {}

    def hf_to_dcp(self, key: str, tensor: torch.Tensor) -> Optional[Tuple[str, torch.Tensor]]:
        """Merge experts; pass ordinary tensors through with Kimi padding fixes."""
        if tensor.dtype not in self._DTYPES.values():
            raise ValueError(f'{key}: unsupported training dtype {tensor.dtype}')
        info = self._expert_info(key)
        if info is not None:
            self._check_expert_shape_dtype(key, info, tensor.shape, tensor.dtype, self._expert_dtypes)
            target = info[3]
            if key in self._buffered_weights or target in self._finished_targets:
                raise ValueError(f'Duplicate HF expert key: {key}')
            self._buffered_weights[key] = tensor
            result = merge_moe_expert_weights(
                self._buffered_weights, self.dcp_to_hf_mapping[target], target,
                **self._EXPERT_KWARGS)
            if result is not None:
                self._finished_targets.add(target)
            return result
        target_shape = self._ordinary_shape(key, tensor.shape)
        self._source_shapes[key] = list(tensor.shape)
        if list(tensor.shape) != target_shape:
            if key.endswith('.self_attn.A_log') and tensor[target_shape[0]:].count_nonzero().item():
                raise ValueError(f'{key}: A_log padding contains nonzero data; refusing to truncate')
            tensor = tensor[:target_shape[0]].clone()
        return key, tensor

    def dcp_to_hf(self, key: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if key in self.dcp_to_hf_mapping:
            tensor = self._to_full(tensor)
            if list(tensor.shape) != self._fused_expert_shape(key, self.num_experts):
                raise ValueError(f'Unexpected fused Kimi expert shape: {key}')
            return split_moe_expert_weights(
                tensor, self.dcp_to_hf_mapping[key], key,
                **self._EXPERT_KWARGS)
        self._ordinary_shape(key, tensor.shape)
        if key.endswith('.self_attn.A_log') and self.linear_heads is not None:
            tensor = self._to_full(tensor)
            source_shape = self._source_shape(key)
            target_shape = self._ordinary_shape(key, source_shape)
            if list(tensor.shape) != target_shape:
                raise ValueError(f'Unexpected trained A_log shape: {key}')
            if source_shape != target_shape:
                restored = tensor.new_zeros(source_shape)
                restored[:self.linear_heads].copy_(tensor)
                tensor = restored
        elif key.endswith(('.block_sparse_moe.gate.weight', '.block_sparse_moe.gate.e_score_correction_bias')):
            source_shape = self._source_shape(key)
            if list(tensor.shape) != source_shape:
                raise ValueError(f'Cannot reconstruct truncated HF router rows for {key}; original shape={source_shape}')
            tensor = self._to_full(tensor)
        return {key: tensor}

    def describe_hf_tensors(self, entries):
        """Validate model metadata and describe ordinary tensors and full expert groups."""
        ordinary, found, dtypes = [], set(), {}
        for key, entry in entries.items():
            if entry['dtype'] not in self._DTYPES:
                raise ValueError(f'{key}: unsupported training dtype {entry["dtype"]}')
            info = self._expert_info(key)
            if info is not None:
                self._check_expert_shape_dtype(key, info, entry['shape'], entry['dtype'], dtypes)
                found.add(key)
            else:
                self._source_shapes[key] = list(entry['shape'])
                shape = self._ordinary_shape(key, entry['shape'])
                if key.endswith('.self_attn.A_log') and list(entry['shape']) != shape:
                    if not (self.hf_dir / entry['file']).resolve().is_relative_to(self.hf_dir):
                        raise ValueError(f'Source tensor path is outside HF directory: {entry["file"]}')
                    # A_log is a small vector; reuse the streaming padding check.
                    self.hf_to_dcp(key, read_tensor(self.hf_dir, entry))
                ordinary.append(dict(key=key, target=key, shape=shape, dtype=entry['dtype']))
        expected = {key for keys in self.hf_to_dcp_mapping for key in keys}
        if found != expected:
            raise ValueError(f'Missing/unexpected Kimi experts: missing={sorted(expected-found)[:6]}, '
                             f'unexpected={sorted(found-expected)[:6]}')
        groups = {}
        for hf_keys, target in self.hf_to_dcp_mapping.items():
            layer = self._expert_info(hf_keys[0])[0]
            stride = len(hf_keys) // self.num_experts
            groups.setdefault(layer, []).append(dict(
                target=target, shape=self._fused_expert_shape(target, self.num_experts),
                dtype=dtypes[layer], keys=[list(hf_keys[i:i+stride]) for i in range(0, len(hf_keys), stride)]))
        return dict(ordinary=sorted(ordinary, key=lambda t: t['key']),
                    expert_groups=[dict(tensors=tensors) for tensors in groups.values()])

    def convert_expert_group(self, target_key, hf_keys, weights, out):
        """Merge one complete EP subgroup using the shared preallocated-output path."""
        if target_key not in self.dcp_to_hf_mapping:
            raise ValueError(f'Unknown Kimi expert target: {target_key}')
        if not isinstance(out, torch.Tensor) or out.dtype not in self._DTYPES.values():
            raise ValueError('Kimi expert output requires a supported floating-point tensor')
        keys = tuple(hf_keys)
        infos = [self._expert_info(key) for key in keys]
        if not keys or any(info is None or info[3] != target_key for info in infos):
            raise ValueError(f'Expert group keys do not match target: {target_key}')
        stride = 1 if target_key.endswith('.down_proj') else 2
        ids = [info[1] for info in infos[::stride]]
        if ids != sorted(set(ids)):
            raise ValueError('Expert group keys must follow ascending expert order')
        if list(out.shape) != self._fused_expert_shape(target_key, len(ids)):
            raise ValueError(f'Expert group output shape mismatch: {target_key}')
        result = merge_moe_expert_weights(
            weights, keys, target_key, out=out, **self._EXPERT_KWARGS)
        if result is None:
            raise ValueError(f'Missing expert group source: {target_key}')
        return result

    def finalize_hf_to_dcp(self):
        """Reject truncated expert streams even when a caller allows partial loading."""
        missing = set(self.dcp_to_hf_mapping) - self._finished_targets
        if self._buffered_weights or missing:
            raise ValueError(f'Incomplete Kimi expert stream: {len(missing)} unfinished projections; '
                             f'buffered keys={list(self._buffered_weights)[:6]}')

    def _expert_info(self, key):
        match = self._EXPERT_RE.fullmatch(key)
        if match is None:
            if '.block_sparse_moe.experts.' in key:
                raise ValueError(f'Unsupported Kimi-K3.1 expert key: {key}')
            return None
        layer, expert = int(match[1]), int(match[2])
        if layer not in self.moe_layers or not 0 <= expert < self.num_experts:
            raise ValueError(f'Unexpected Kimi-K3.1 expert key: {key}')
        part = match[3]
        projection = 'down_proj' if part == 'w2' else 'gate_up_proj'
        return layer, expert, part, f'language_model.model.layers.{layer}.block_sparse_moe.experts.{projection}'

    def _ordinary_shape(self, key, shape):
        shape = list(shape)
        if key.endswith(('.block_sparse_moe.gate.weight', '.block_sparse_moe.gate.e_score_correction_bias')):
            expected = [self.num_experts, self.hidden_size] if key.endswith('.gate.weight') else [self.num_experts]
            if shape != expected:
                raise ValueError(f'{key}: incompatible router shape {shape}; expected {expected}')
            return shape
        if key.endswith('.self_attn.A_log') and self.linear_heads is not None:
            padded = 1 << (self.linear_heads - 1).bit_length()
            if shape not in ([self.linear_heads], [padded]):
                raise ValueError(f'{key}: incompatible A_log shape {shape}')
            return [self.linear_heads]
        return shape

    def _fused_expert_shape(self, target_key, expert_count):
        tail = ([self.intermediate_size, self.expert_hidden_size] if target_key.endswith('.down_proj')
                else [self.expert_hidden_size, 2 * self.intermediate_size])
        return [expert_count] + tail

    def _check_expert_shape_dtype(self, key, info, shape, dtype, dtypes):
        layer, _, part, _ = info
        expected = ([self.expert_hidden_size, self.intermediate_size] if part == 'w2'
                    else [self.intermediate_size, self.expert_hidden_size])
        if list(shape) != expected:
            raise ValueError(f'{key}: expected expert shape {expected}, got {list(shape)}')
        if dtypes.setdefault(layer, dtype) != dtype:
            raise ValueError(f'Layer {layer}: mixed expert dtypes cannot be fused losslessly')

    def _source_shape(self, key):
        if key not in self._source_shapes:
            self._source_shapes[key] = read_hf_tensor_shape(self.hf_dir, key, self._source_metadata)
        return self._source_shapes[key]


class Qwen35WeightTransformPipeline(WeightTransformPipeline):
    """Weight transform pipeline for Qwen3.5.
    """
    def __init__(
        self,
        hf_dir: str,
        mtp_num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        self.expert_weight_patterns = [
            r"model\.language_model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
            r"model\.language_model\.layers\.\d+\.mlp\.experts\.down_proj",
            r"mtp\.layers\.\d+\.mlp\.experts\.gate_up_proj",
            r"mtp\.layers\.\d+\.mlp\.experts\.down_proj",
        ]
        self._buffered_weights: Dict[str, torch.Tensor] = {}

        num_experts = getattr(text_config, "num_experts", 0)
        mtp_num_hidden_layers = getattr(text_config, "mtp_num_hidden_layers", 0)
        if mtp_num_layers is not None:
            mtp_num_hidden_layers = min(mtp_num_hidden_layers, mtp_num_layers)
        for layer in range(mtp_num_hidden_layers if num_experts > 0 else 0):
            weight_path = f"mtp.layers.{layer}.mlp.experts"
            gate_up_hf_keys = tuple(
                f"{weight_path}.{expert}.{projection}.weight"
                for expert in range(num_experts)
                for projection in ("gate_proj", "up_proj")
            )
            down_hf_keys = tuple(
                f"{weight_path}.{expert}.down_proj.weight"
                for expert in range(num_experts)
            )
            self.hf_to_dcp_mapping[gate_up_hf_keys] = f"{weight_path}.gate_up_proj"
            self.hf_to_dcp_mapping[down_hf_keys] = f"{weight_path}.down_proj"

        self.dcp_to_hf_mapping = {
            dcp_key: hf_keys
            for hf_keys, dcp_key in self.hf_to_dcp_mapping.items()
        }

    def hf_to_dcp(
        self, key: str, tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Convert a Huggingface weight to DCP format.

        Shape symbols:
            e: Expert index.
            E: Number of experts.
            H: Hidden size.
            I: Intermediate size.
        """
        # Merge per-expert MTP weights:
        # experts.{e}.{gate,up}_proj.weight [I, H] -> experts.gate_up_proj [E, H, 2I]
        # experts.{e}.down_proj.weight [H, I] -> experts.down_proj [E, I, H]
        for hf_keys, dcp_key in self.hf_to_dcp_mapping.items():
            if key not in hf_keys:
                continue
            self._buffered_weights[key] = tensor
            return merge_moe_expert_weights(
                state_dict=self._buffered_weights,
                hf_keys=hf_keys,
                dcp_key=dcp_key,
            )

        # Permute fused MoE weights:
        # experts.gate_up_proj [E, 2I, H] -> [E, H, 2I]
        # experts.down_proj [E, H, I] -> [E, I, H]
        tensor = permute_moe_expert(key, tensor, self.expert_weight_patterns)
        return key, tensor

    def dcp_to_hf(
        self, key: str, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert a DCP weight to Huggingface format.

        Shape symbols:
            e: Expert index.
            E: Number of experts.
            H: Hidden size.
            I: Intermediate size.
        """
        # Split merged MTP weights:
        # experts.gate_up_proj [E, H, 2I] -> experts.{e}.{gate,up}_proj.weight [I, H]
        # experts.down_proj [E, I, H] -> experts.{e}.down_proj.weight [H, I]
        hf_keys = self.dcp_to_hf_mapping.get(key)
        if hf_keys is not None:
            # Gather before unbinding the sharded expert dimension.
            return split_moe_expert_weights(
                tensor=self._to_full(tensor),
                hf_keys=hf_keys,
                dcp_key=key,
            )

        # Permute fused MoE weights:
        # experts.gate_up_proj [E, H, 2I] -> [E, 2I, H]
        # experts.down_proj [E, I, H] -> [E, H, I]
        tensor = permute_moe_expert(key, tensor, self.expert_weight_patterns)
        return {key: tensor}

class Glm53FlashWeightTransformPipeline(WeightTransformPipeline):
    RENAME_RULES = [
        (re.compile(r"\.hc_attn_(base|fn|scale)$"), r".attn_hc.\1"),
        (re.compile(r"\.hc_ffn_(base|fn|scale)$"), r".ffn_hc.\1"),
        (re.compile(r"\.self_attn\.(A_log|dt_bias)$"), r".self_attn.forget_gate.\1"),
        (re.compile(r"\.self_attn\.(f_a_proj|f_b_proj)\."), r".self_attn.forget_gate.\1."),
    ]
    RENAME_RULES_REVERSED = [
        (re.compile(r"\.attn_hc\.(base|fn|scale)$"), r".hc_attn_\1"),
        (re.compile(r"\.ffn_hc\.(base|fn|scale)$"), r".hc_ffn_\1"),
        (re.compile(r"\.self_attn\.forget_gate\.(A_log|dt_bias)$"), r".self_attn.\1"),
        (re.compile(r"\.self_attn\.forget_gate\.(f_a_proj|f_b_proj)\."), r".self_attn.\1."),
    ]
    CONV_ORDER = ("q", "k", "v")
    CONV_RE = re.compile(r"^(?P<prefix>.*\.self_attn)\.(?P<which>[qkv])_conv1d\.weight$")
    QKV_ORDER = ("q", "k", "v")
    QKV_RE = re.compile(r"^(?P<prefix>.*\.self_attn)\.(?P<which>[qkv])_proj\.weight$")
    MLP_ORDER = ("gate", "up")
    MLP_RE = re.compile(r"^(?P<prefix>.*)\.(?P<which>gate|up)_proj\.weight$")

    def __init__(self, hf_dir: str, mtp_num_layers: Optional[int] = None) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)

        num_experts = getattr(text_config, "n_routed_experts", 0)
        mlp_layer_types = getattr(text_config, "mlp_layer_types", None) or []
        num_layers = getattr(text_config, "num_hidden_layers", len(mlp_layer_types))

        for layer_idx, mlp_type in enumerate(mlp_layer_types[:num_layers]):
            if mlp_type != "sparse" or num_experts <= 0:
                continue
            weight_path = f"model.language_model.layers.{layer_idx}.mlp.experts"
            gate_up_hf_keys = tuple(
                f"{weight_path}.{expert}.{projection}.weight"
                for expert in range(num_experts)
                for projection in ("gate_proj", "up_proj")
            )
            down_hf_keys = tuple(
                f"{weight_path}.{expert}.down_proj.weight" for expert in range(num_experts)
            )
            self.hf_to_dcp_mapping[gate_up_hf_keys] = f"{weight_path}.gate_up_proj"
            self.hf_to_dcp_mapping[down_hf_keys] = f"{weight_path}.down_proj"

        self.dcp_to_hf_mapping = {
            dcp_key: hf_keys for hf_keys, dcp_key in self.hf_to_dcp_mapping.items()
        }

        self._buffered_weights: Dict[str, torch.Tensor] = {}
        self._buffered_conv: Dict[str, Dict[str, torch.Tensor]] = {}
        layer_types = getattr(text_config, "layer_types", None) or []
        self._qkv_prefixes = {
            f"model.language_model.layers.{layer_idx}.self_attn"
            for layer_idx, layer_type in enumerate(layer_types[:num_layers])
            if layer_type == "linear_attention"
        }
        self._buffered_qkv: Dict[str, Dict[str, torch.Tensor]] = {}
        self._mlp_prefixes = {
            f"model.language_model.layers.{layer_idx}.mlp"
            + ("" if mlp_type != "sparse" else ".shared_experts")
            for layer_idx, mlp_type in enumerate(mlp_layer_types[:num_layers])
        }
        self._buffered_mlp: Dict[str, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _rename(key: str, rules) -> str:
        for pattern, replacement in rules:
            new_key, n = pattern.subn(replacement, key)
            if n:
                return new_key
        return key

    def hf_to_dcp(
        self, key: str, tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Convert a Huggingface weight to DCP format.
        Shape symbols:
            e: Expert index.  E: Number of experts.
            H: Hidden size.   I: Intermediate size.  C: Conv kernel size.
        """
        # 1. experts.{e}.{gate,up}_proj.weight [I, H] -> experts.gate_up_proj [E, 2I, H]
        #    experts.{e}.down_proj.weight     [H, I] -> experts.down_proj    [E, H, I]
        for hf_keys, dcp_key in self.hf_to_dcp_mapping.items():
            if key not in hf_keys:
                continue
            self._buffered_weights[key] = tensor
            return merge_moe_expert_weights(
                state_dict=self._buffered_weights,
                hf_keys=hf_keys,
                dcp_key=dcp_key,
                transpose=False,
            )
        # 2. qkv: Concatenates three [qkv_dim, H] tensors into [3 * qkv_dim, H] in the order of q, k, v.
        match = self.QKV_RE.match(key)
        if match and match.group("prefix") in self._qkv_prefixes:
            prefix = match.group("prefix")
            group = self._buffered_qkv.setdefault(prefix, {})
            group[match.group("which")] = tensor
            if len(group) < len(self.QKV_ORDER):
                return None
            merged = torch.cat([group[which] for which in self.QKV_ORDER], dim=0)
            del self._buffered_qkv[prefix]
            return f"{prefix}.qkv_proj.weight", merged
        # 3. mlp gate/up: Concatenates two [I, H] tensors into [2I, H] in the order of gate, up.
        match = self.MLP_RE.match(key)
        if match and match.group("prefix") in self._mlp_prefixes:
            prefix = match.group("prefix")
            group = self._buffered_mlp.setdefault(prefix, {})
            group[match.group("which")] = tensor
            if len(group) < len(self.MLP_ORDER):
                return None
            merged = torch.cat([group[which] for which in self.MLP_ORDER], dim=0)
            del self._buffered_mlp[prefix]
            return f"{prefix}.gate_up_proj.weight", merged

        # 4. conv1d: Concatenates three [qkv_dim, 1, C] tensors into [3 * qkv_dim, 1, C] in the order of q, k, v.
        match = self.CONV_RE.match(key)
        if match:
            prefix = match.group("prefix")
            group = self._buffered_conv.setdefault(prefix, {})
            group[match.group("which")] = tensor
            if len(group) < len(self.CONV_ORDER):
                return None
            merged = torch.cat([group[which] for which in self.CONV_ORDER], dim=0)
            del self._buffered_conv[prefix]
            return f"{prefix}.conv1d.weight", merged

        # 5. rename only
        return self._rename(key, self.RENAME_RULES), tensor
    def dcp_to_hf(self, key: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert a DCP weight to Huggingface format. 是 hf_to_dcp 的逆。"""
        hf_keys = self.dcp_to_hf_mapping.get(key)
        if hf_keys is not None:
            return split_moe_expert_weights(
                tensor=self._to_full(tensor), hf_keys=hf_keys, dcp_key=key, transpose=False
            )
        if key.endswith(".self_attn.qkv_proj.weight"):
            prefix = key[: -len(".qkv_proj.weight")]
            chunks = torch.chunk(self._to_full(tensor), len(self.QKV_ORDER), dim=0)
            return {
                f"{prefix}.{which}_proj.weight": chunk.contiguous()
                for which, chunk in zip(self.QKV_ORDER, chunks)
            }
        if key.endswith(".gate_up_proj.weight"):
            prefix = key[: -len(".gate_up_proj.weight")]
            chunks = torch.chunk(self._to_full(tensor), len(self.MLP_ORDER), dim=0)
            return {
                f"{prefix}.{which}_proj.weight": chunk.contiguous()
                for which, chunk in zip(self.MLP_ORDER, chunks)
            }
        if key.endswith(".self_attn.conv1d.weight"):
            prefix = key[: -len(".conv1d.weight")]
            chunks = torch.chunk(self._to_full(tensor), len(self.CONV_ORDER), dim=0)
            return {
                f"{prefix}.{which}_conv1d.weight": chunk.contiguous()
                for which, chunk in zip(self.CONV_ORDER, chunks)
            }

        return {self._rename(key, self.RENAME_RULES_REVERSED): tensor}

class DiffusersKeyMapTransformPipeline(WeightTransformPipeline):
    """Generic diffusers -> native per-tensor transform pipeline driven by
    subclass key maps (``CONVERT_MAPPING`` / ``STR_REPLACE_MAPPING``).

    New models only ship the mapping tables and register a subclass in
    ``WEIGHT_TRANSFORM_PIPELINES`` (e.g. ``Wan22DiffusersTransformPipeline``);
    the checkpoint layer carries no model-specific knowledge. Saving back to
    diffusers format is not supported.
    """

    CONVERT_MAPPING: Dict[str, str] = {}
    STR_REPLACE_MAPPING: Dict[str, str] = {}

    def __init__(self, hf_dir: Optional[str] = None, **kwargs) -> None:
        # ``hf_dir``/``mtp_num_layers`` are accepted (and ignored) so the class
        # can be built through ``build_weight_transform`` / ``HFCheckpointer``
        # with the same call signature as config-driven pipelines.
        super().__init__()

    def hf_to_dcp(
        self, key: str, tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Rename one diffusers key to the native layout; returns ``None`` for
        dropped tensors (e.g. diffusers-internal ``_extra_state`` entries)."""
        key = convert_diffusers_key(key, self.CONVERT_MAPPING, self.STR_REPLACE_MAPPING)
        if key is None:
            return None
        return key, tensor

    def dcp_to_hf(
        self, key: str, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Saving weights in diffusers format is not supported."
        )


# Wan2.2 diffusers -> native WanModel key mapping tables.

# Exact key renames (old_key -> new_key)
WAN22_DIFFUSERS_CONVERT_MAPPING = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
}

# Sub-string replacements applied to every key
WAN22_DIFFUSERS_STR_REPLACE_MAPPING = {
    "attn1.norm_q": "self_attn.norm_q",
    "attn1.norm_k": "self_attn.norm_k",
    "attn2.norm_q": "cross_attn.norm_q",
    "attn2.norm_k": "cross_attn.norm_k",
    "attn1.to_q.": "self_attn.q.",
    "attn1.to_k.": "self_attn.k.",
    "attn1.to_v.": "self_attn.v.",
    "attn1.to_out.0.": "self_attn.o.",
    "attn2.to_q.": "cross_attn.q.",
    "attn2.to_k.": "cross_attn.k.",
    "attn2.to_v.": "cross_attn.v.",
    "attn2.to_out.0.": "cross_attn.o.",
    ".ffn.net.0.proj.": ".ffn.0.",
    ".ffn.net.2.": ".ffn.2.",
    "scale_shift_table": "modulation",
    ".norm2.": ".norm3.",
}


class Wan22DiffusersTransformPipeline(DiffusersKeyMapTransformPipeline):
    """Wan2.2 diffusers -> native transform pipeline, driven by the tables above."""

    CONVERT_MAPPING = WAN22_DIFFUSERS_CONVERT_MAPPING
    STR_REPLACE_MAPPING = WAN22_DIFFUSERS_STR_REPLACE_MAPPING


class MagiHumanWeightTransformPipeline(WeightTransformPipeline):
    """Weight transform pipeline for MagiHuman.

    MagiHuman trains with the gated ``up_gate_proj`` rows grouped into two
    contiguous halves, while the public checkpoint interleaves them. Only the
    layout differs: keys, shapes and dtypes are identical on both sides, so this
    pipeline never renames or buffers a tensor.
    """

    def __init__(
        self,
        hf_dir: str,
        mtp_num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Imported lazily: the model package triggers `@model_register` on import,
        # which must not be a side effect of loading this checkpoint module.
        from mindspeed_mm.fsdp.models.magihuman.magihuman_fsdp2.weight_layout import (
            convert_up_gate_checkpoint_tensor,
        )

        self._convert = convert_up_gate_checkpoint_tensor

    def hf_to_dcp(
        self, key: str, tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """De-interleave gated ``up_gate_proj`` rows into contiguous halves."""
        return key, self._convert(key, tensor, to_internal=True)

    def dcp_to_hf(
        self, key: str, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Restore the upstream interleaved ``up_gate_proj`` row order."""
        return {key: self._convert(key, tensor, to_internal=False)}


WEIGHT_TRANSFORM_PIPELINES = {
    "kimi_k31_pro": KimiK31WeightTransformPipeline,
    "qwen3_5_moe": Qwen35WeightTransformPipeline,
    "wan2_2": Wan22DiffusersTransformPipeline,
    "glm5_next": Glm53FlashWeightTransformPipeline,
    "magihuman": MagiHumanWeightTransformPipeline,
}
# Note: ``DiffusersKeyMapTransformPipeline`` is a base class with empty mapping
# tables and must NOT be registered directly; new diffusers-source models add a
# concrete subclass with their key tables and register it here keyed by model_id.
