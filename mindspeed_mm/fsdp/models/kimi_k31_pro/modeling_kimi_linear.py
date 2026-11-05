from functools import wraps

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from packaging import version
from torch import nn
from transformers.cache_utils import Cache

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.models.kimi_k3 import modeling_kimi_linear as _k3_linear
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

from .configuration_kimi_k3 import KimiLinearConfig

if IS_NPU_AVAILABLE:
    import torch_npu


class KimiDSAIndexer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        dsa_config = config.dsa_config
        self.layer_idx = layer_idx
        self.n_heads = dsa_config["index_num_heads"]
        self.head_dim = dsa_config["index_head_dim"]
        self.index_topk = dsa_config["index_topk"]
        self.dsa_implementation = getattr(config, "dsa_implementation", "eager")
        if dsa_config.get("index_rope_dim", 0) != 0:
            raise NotImplementedError("Kimi-K3.1-Pro only supports index_rope_dim=0.")
        if dsa_config.get("index_k_norm_type", "layernorm") != "layernorm":
            raise NotImplementedError("Kimi-K3.1-Pro only supports layernorm in the DSA indexer.")
        if dsa_config.get("index_topk_freq", 1) != 1:
            raise NotImplementedError("Kimi-K3.1-Pro requires index_topk_freq=1.")

        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states))
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype))
        weights = weights * (self.n_heads ** -0.5)
        sparse_count = min(self.index_topk, seq_len)

        if self.dsa_implementation == "ascendc":
            if not IS_NPU_AVAILABLE:
                raise RuntimeError("dsa_implementation='ascendc' requires an NPU.")
            actual_seq_lengths = None
            if padding_mask is not None:
                actual_seq_lengths = padding_mask.sum(dim=-1, dtype=torch.int32)
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                q.contiguous(),
                k.unsqueeze(2).contiguous(),
                (weights * self.softmax_scale).to(q.dtype).contiguous(),
                actual_seq_lengths_query=actual_seq_lengths,
                actual_seq_lengths_key=actual_seq_lengths,
                layout_query="BSND",
                layout_key="BSND",
                sparse_count=sparse_count,
                sparse_mode=3,
                return_value=False,
            )
            return topk_indices.squeeze(2)

        scores = torch.matmul(q.float(), k.float().unsqueeze(1).transpose(-1, -2))
        scores = F.relu(scores * self.softmax_scale)
        index_scores = torch.matmul(weights.float().unsqueeze(-2), scores).squeeze(-2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)
        key_positions = torch.arange(seq_len, device=hidden_states.device)
        invalid = key_positions[None, None, :] > position_ids[:, :, None]
        if padding_mask is not None:
            invalid = invalid | ~padding_mask.bool()[:, None, :]
        index_scores = index_scores.masked_fill(invalid, float("-inf"))
        return index_scores.topk(sparse_count, dim=-1).indices.to(torch.int32)


class KimiDSAAttention(_k3_linear.KimiMLAAttention):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.dsa_implementation = getattr(config, "dsa_implementation", "eager")
        if self.dsa_implementation not in ("eager", "ascendc"):
            raise ValueError(
                f"Unsupported dsa_implementation: {self.dsa_implementation}. "
                "Expected 'eager' or 'ascendc'."
            )
        if config.dsa_config is None:
            raise ValueError("KimiDSAAttention requires dsa_config.")
        self.indexer = KimiDSAIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise NotImplementedError("The Kimi-K3.1-Pro training adapter does not support the DSA KV cache.")

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pass = self.kv_a_layernorm(kv_pass)
        k_rot = k_rot.view(batch_size, seq_length, 1, self.qk_rope_head_dim)

        ps = get_parallel_state() if dist.is_initialized() else None
        if ps is not None and ps.is_ulysses_enable():
            raise NotImplementedError("Kimi-K3.1-Pro DSA requires ulysses_parallel_size=1.")

        padding_mask = kwargs.get("padding_mask")
        if padding_mask is None and attention_mask is not None:
            if attention_mask.ndim == 2:
                padding_mask = attention_mask
            elif attention_mask.ndim == 4:
                visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask >= 0
                padding_mask = visible[:, 0].any(dim=1)
            else:
                raise ValueError("DSA attention_mask must be a 2-D padding mask or a 4-D causal mask.")

        topk_indices = None
        if seq_length > self.indexer.index_topk:
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                padding_mask=padding_mask,
                position_ids=position_ids,
            )
        if self.dsa_implementation == "ascendc" and topk_indices is not None:
            attn_output = self._ascendc_attention(q_pass, q_rot, k_pass, k_rot, topk_indices)
        else:
            attn_output = self._eager_attention(
                hidden_states, q_pass, q_rot, k_pass, k_rot, topk_indices,
                position_ids, padding_mask
            )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            attn_output = attn_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_output)

    def _eager_attention(
        self,
        hidden_states: torch.Tensor,
        q_pass: torch.Tensor,
        q_rot: torch.Tensor,
        k_pass: torch.Tensor,
        k_rot: torch.Tensor,
        topk_indices: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        key_shape = (
            batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        key_value = self.kv_b_proj(k_pass).view(key_shape).transpose(1, 2)
        key_nope, value_states = torch.split(
            key_value, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        key_rot = k_rot.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((key_nope, key_rot), dim=-1)

        if topk_indices is None:
            selected = torch.ones(
                batch_size, seq_length, seq_length, dtype=torch.bool, device=hidden_states.device
            )
        else:
            selected = torch.zeros(
                batch_size, seq_length, seq_length, dtype=torch.bool, device=hidden_states.device
            ).scatter(-1, topk_indices.long(), True)
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=hidden_states.device).expand(batch_size, -1)
        key_positions = torch.arange(seq_length, device=hidden_states.device)
        selected &= key_positions[None, None, :] <= position_ids[:, :, None]
        if padding_mask is not None:
            selected &= padding_mask.bool()[:, None, :]
        sparse_mask = hidden_states.new_zeros((batch_size, 1, seq_length, seq_length))
        sparse_mask = sparse_mask.masked_fill(
            ~selected.unsqueeze(1), torch.finfo(hidden_states.dtype).min
        )
        attn_output, _ = _k3_linear.eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            sparse_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        return attn_output

    def _ascendc_attention(
        self,
        q_pass: torch.Tensor,
        q_rot: torch.Tensor,
        k_pass: torch.Tensor,
        k_rot: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        from mindspeed_mm.fsdp.ops.dsa.sfa_ascendc_kimi import sparse_flash_attention

        weight = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_uk = weight[:, :self.qk_nope_head_dim, :]
        w_uv = weight[:, self.qk_nope_head_dim:, :]
        query = torch.einsum(
            "bshd,hdc->bshc", q_pass.transpose(1, 2), w_uk.to(q_pass.dtype)
        )
        latent = k_pass.unsqueeze(2)
        attn_output = sparse_flash_attention(
            query,
            latent,
            latent,
            topk_indices,
            scale=self.scaling,
            rope_head_dim=self.qk_rope_head_dim,
            query_rope=q_rot.transpose(1, 2).contiguous(),
            key_rope=k_rot.contiguous(),
        )
        return torch.einsum("bshc,hvc->bshv", attn_output, w_uv.to(attn_output.dtype))


class KimiK31DecoderLayer(_k3_linear.KimiDecoderLayer):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = _k3_linear.KimiDeltaAttention(config=config, layer_idx=layer_idx)
        elif config.is_mla:
            self.is_linear_attn = False
            dsa_config = getattr(config, "dsa_config", None)
            is_dsa_layer = dsa_config is not None and (
                layer_idx + 1
            ) in dsa_config.get("dsa_layers", [])
            attention_cls = KimiDSAAttention if is_dsa_layer else _k3_linear.KimiMLAAttention
            self.self_attn = attention_cls(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError

        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = _k3_linear.KimiSparseMoeBlock(config)
        else:
            self.mlp = _k3_linear.KimiMLP(config)
        self.input_layernorm = _k3_linear.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _k3_linear.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = _k3_linear.KimiRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = _k3_linear.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)


def apply_kimi_k31_dsa_patch():
    if getattr(_k3_linear, "_kimi_k31_dsa_patched", False):
        return

    _k3_linear.KimiDecoderLayer = KimiK31DecoderLayer
    linear_model_forward = _k3_linear.KimiLinearModel.forward

    @wraps(linear_model_forward)
    def forward_with_padding_mask(self, *args, **kwargs):
        if "attention_mask" in kwargs and "padding_mask" not in kwargs:
            kwargs["padding_mask"] = kwargs["attention_mask"]
        return linear_model_forward(self, *args, **kwargs)

    _k3_linear.KimiLinearModel.forward = forward_with_padding_mask
    if version.parse(transformers.__version__) >= version.parse("5.0.0"):
        create_causal_mask = _k3_linear.create_causal_mask

        def create_causal_mask_compat(*args, **kwargs):
            if "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return create_causal_mask(*args, **kwargs)

        _k3_linear.create_causal_mask = create_causal_mask_compat
    _k3_linear._kimi_k31_dsa_patched = True


__all__ = ["KimiDSAAttention", "KimiDSAIndexer", "apply_kimi_k31_dsa_patch"]
