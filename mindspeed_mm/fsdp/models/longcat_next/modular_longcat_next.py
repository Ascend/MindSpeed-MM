import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerNorm as RMSNorm


class FlashVarLenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False, window_size=(-1, -1)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal
        self.window_size = window_size

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, self.num_heads, self.head_dim).contiguous()
        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(bsz, self.num_heads, self.head_dim).contiguous()
        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, self.num_heads, self.head_dim).contiguous()

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()

        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_len, cu_len, max_seqlen,
                                             max_seqlen, causal=self.causal,
                                             window_size=self.window_size)  # (bsz * qlen, nheads, headdim)
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CasualDepthTransformerLayer(nn.Module):
    def __init__(self, depth, transformer_dim, transformer_ffn_scale):
        super().__init__()
        self.depth = depth
        self.transformer_dim = transformer_dim
        self.transformer_ffn_scale = transformer_ffn_scale
        self.num_heads = self.transformer_dim // 128

        if self.transformer_dim % 128 != 0:
            raise AssertionError("transformer_dim % 128 needs to be 0")
        if self.transformer_dim % depth != 0:
            raise AssertionError("transformer_dim % depth needs to be 0")

        self.self_attention = FlashVarLenAttention(embed_dim=self.transformer_dim, num_heads=self.num_heads,
                                                   causal=True)

        self.layernorm1 = RMSNorm(self.transformer_dim)
        self.layernorm2 = RMSNorm(self.transformer_dim)

        self.linear1 = nn.Linear(self.transformer_dim, self.transformer_ffn_scale * self.transformer_dim)
        self.linear2 = nn.Linear(self.transformer_ffn_scale * self.transformer_dim, self.transformer_dim)

    def forward(self, x):
        bsz = x.shape[0]
        res = x
        x = self.layernorm1(x)
        seqlens = torch.tensor([self.depth] * bsz, dtype=torch.int32, device=x.device)
        _x = self.self_attention(x.view(-1, self.transformer_dim), seqlens)
        _x = _x.view(bsz, self.depth, self.transformer_dim).contiguous()

        _res = _x + res  # (bs, sl, d)
        res = self.layernorm2(_res)
        x = torch.einsum('bld,tld->blt', res, torch.reshape(self.linear1.weight,
                                                            (self.transformer_ffn_scale * self.transformer_dim // self.depth,
                                                             self.depth, self.transformer_dim)))
        x = torch.nn.functional.gelu(x)
        x = torch.einsum('blt,dlt->bld', x, torch.reshape(self.linear2.weight, (self.transformer_dim, self.depth,
                                                                                self.transformer_ffn_scale * self.transformer_dim // self.depth)))
        return _res + x


class CasualDepthTransformerHead(nn.Module):
    """
    Depth-wise causal transformer head shared by image/audio heads.
    """

    def __init__(
            self,
            hidden_size,
            codebook_sizes,
            transformer_layer_num,
            transformer_dim,
            transformer_ffn_scale,
            gradient_checkpointing=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.codebook_sizes = codebook_sizes
        self.transformer_ffn_scale = transformer_ffn_scale
        self.gradient_checkpointing = gradient_checkpointing

        if self.transformer_ffn_scale > 0:
            self.hidden_norm = RMSNorm(self.hidden_size)
            self.hidden_proj = nn.Linear(self.hidden_size, transformer_dim, bias=False)

        self.transformer_layers = nn.ModuleList(
            [
                CasualDepthTransformerLayer(len(codebook_sizes), transformer_dim, transformer_ffn_scale)
                for _ in range(transformer_layer_num)
            ]
        )
        self.headnorm = RMSNorm(transformer_dim)
        self.heads = nn.ModuleList(
            [nn.Linear(transformer_dim, vq_size + 1) for vq_size in codebook_sizes]
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, visual_tokens, visual_emb_layers, level):
        main_device = x.device
        visual_tokens = visual_tokens.to(main_device)
        visual_emb_layers = visual_emb_layers.to(x.device)

        cumsum_visual_embed = torch.stack([
            visual_emb_layers(visual_tokens[..., i])
            for i, vq_size in enumerate(self.codebook_sizes[:-1])
        ], dim=1).to(x.device)

        cumsum_visual_embed = torch.cumsum(cumsum_visual_embed, dim=1)  # (bs, depth-1, d)

        hidden_states = torch.concat([x.reshape(-1, 1, self.hidden_size), cumsum_visual_embed], dim=1)  # (bs, depth, d)
        if hidden_states.size(1) != len(self.codebook_sizes):
            raise AssertionError("hidden_states.size(1) should equal with len(self.codebook_sizes)")

        if self.transformer_ffn_scale > 0:
            hidden_states = self.hidden_norm(hidden_states)
            hidden_states = self.hidden_proj(hidden_states)

        for i, tlayer in enumerate(self.transformer_layers):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), hidden_states,
                )
            else:
                hidden_states = tlayer(
                    hidden_states,
                )
        hidden_states = self.headnorm(hidden_states)
        logits = self.heads[level](hidden_states[:, level])
        return logits
