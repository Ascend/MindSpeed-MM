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

"""Native H3 to head-interleaved QKV and gate/up LoRA conversion."""

import torch


def convert_native_h3_lora(pairs, head_dim):
    if head_dim < 1:
        raise ValueError("head_dim must be positive")
    result, used = {}, set()

    def path(name):
        return name.replace("transformer_blocks.", "blocks.", 1).replace(
            "token_refiner.refiner_blocks.", "token_refiner.blocks.", 1
        )

    def add(name, a, b):
        if name in result:
            raise ValueError(f"Duplicate converted module: {name}")
        result[name] = {"A": a.contiguous(), "B": b.contiguous()}

    attentions = sorted(
        {name.rsplit(".", 1)[0] for name in pairs if name.endswith((".attn.to_q", ".attn.to_k", ".attn.to_v"))}
    )
    for attention in attentions:
        group = [
            (index, attention + ".to_" + side) for index, side in enumerate("qkv") if attention + ".to_" + side in pairs
        ]
        first = pairs[group[0][1]]
        out, inputs = first["B"].shape[0], first["A"].shape[1]
        if out % head_dim or any(
            pairs[name]["B"].shape[0] != out or pairs[name]["A"].shape[1] != inputs for _, name in group
        ):
            raise ValueError(f"Inconsistent QKV dimensions/head_dim: {attention}")
        shared = all(torch.equal(first["A"], pairs[name]["A"]) for _, name in group)
        a = first["A"] if shared else torch.cat([pairs[name]["A"] for _, name in group])
        b = first["B"].new_zeros(out * 3, a.shape[0])
        offset = 0
        for index, name in group:
            rank = pairs[name]["A"].shape[0]
            b.view(out // head_dim, 3, head_dim, a.shape[0])[:, index, :, offset : offset + rank] = pairs[name][
                "B"
            ].view(out // head_dim, head_dim, rank)
            offset += 0 if shared else rank
            used.add(name)
        add(path(attention) + ".qkv_proj", a, b)
    for name in sorted(set(pairs) - used):
        a, b = pairs[name]["A"], pairs[name]["B"]
        target = path(name)
        if target.endswith(".attn.to_out.0"):
            target = target[: -len("to_out.0")] + "out_proj"
        elif target.endswith(".ff.net.0.proj"):
            if b.shape[0] % 2:
                raise ValueError(f"Odd gate/up size: {name}")
            up, gate = b.chunk(2)
            b = torch.cat((gate, up))
            target = target[: -len("ff.net.0.proj")] + "mlp.fc1"
        elif target.endswith(".ff.net.2"):
            target = target[: -len("ff.net.2")] + "mlp.fc2"
        else:
            raise ValueError(f"Unsupported native H3 target: {name}")
        add(target, a, b)
    return result
