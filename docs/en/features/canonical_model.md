# Canonical Models

## Problem Analysis

The implementation logic of the Qwen-VL series models under the Megatron framework differs significantly from the mainstream implementation in Hugging Face Transformers. This discrepancy not only leads to considerable computational deviations in scenarios such as LoRA fine-tuning but also creates difficulties in model migration and adaptation between different frameworks.

## Implementation Differences

Megatron performs fusion and interleaved reordering operations on the model's core modules. Taking the calculation of the q, k, and v matrices in Qwen2.5-VL as an example, the key differences from the standard Transformers implementation are as follows.

### QKV Calculation Logic in the Attention Layer

* Standard Transformers Implementation: Input `hidden_states` into the independent `q_proj`, `k_proj`, and `v_proj` layers separately to directly obtain the q, k, and v matrices.
* Megatron Implementation: The weight matrices of the original model's `q_proj`, `k_proj`, and `v_proj` layers are split and rearranged, then fused into a single `linear_qkv` layer. After `hidden_states` is input into this layer, a fused `qkv` output tensor is first obtained, which is then split and reordered to yield the q, k, and v matrices.
![QKV calculation implementation differences under Megatron](../../../sources/images/canonical_model/img.png)

### FC1 Calculation Logic in the MLP Layer

Megatron also fuses `gate_proj` and `up_proj` in the MLP layer into a single `linear_fc1` layer, which is inconsistent with the standard layered implementation logic of Transformers.

### Differences in LoRA Fine-tuning Scenarios

The fusion operations performed by Megatron on the above modules lead to a mismatch in parameter scale compared to the standard Transformers implementation in LoRA fine-tuning scenarios. For example, the LoRA-A matrix for the qkv layer has only one-third the parameter count of the standard implementation, resulting in algorithmic nonequivalence. Consequently, LoRA weights trained in the two frameworks cannot be compatibly converted or loaded across frameworks.

![1](../../../sources/images/canonical_model/img_1.png)

## Solution

For modules optimized with fusion and interleaving in the Megatron framework, MindSpeed MM provides an adaptation solution equivalent to the standard Transformers implementation, eliminating calculation differences caused by model structure discrepancies between different frameworks and resolving cross-framework switching incompatibility issues. Currently supported models include Qwen2.5-VL and VideoAlign.

## How to Use

Taking Qwen2.5-VL as an example, add `canonical_model` in `model_xxb.json` and enable it:

```json
{
  "model_id": "qwen2_5vl",
  "img_context_token_id": 151655,
  "vision_start_token_id": 151652,
  "image_encoder": {
    "vision_encoder": {
      "model_id": "qwen2vit",
      "canonical_model": true,  // Enable the canonical equivalent implementation of the vision encoder
      ...
    },
    ...
    "text_decoder": {
      "model_id": "qwen2lm",
      "canonical_model": true,  // Enable the canonical equivalent implementation of the text decoder
      ...
    }
  }
}
```
