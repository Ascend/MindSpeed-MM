# Online Data Rearrangement

## Technical Background

A training sample for a multimodal large model is a sequence composed of interleaved multimodal tokens such as text, images, and audio. The number of tokens for different modality samples varies significantly and changes dynamically (e.g., dynamic resolution). This leads to variations in the computational load across different encoders and the backbone network, causing load imbalance. Specific issues include `intra-microbatch` (across DP ranks) and `inter-microbatch` (within a DP rank) imbalance.

To address the data heterogeneity issue, MindSpeed MM has designed a packing + online data rearrangement solution.

## Packing + Online Data Rearrangement

### Solution Introduction

- Objective:Balance the computational load across DP ranks for the LLM (computational load defined as the sum of squares of sub-sequence lengths, i.e., `sub_seq ** 2`).
- Constraint: Perform sequence packing and concatenation based on the maximum sequence length (`max_seq_len`).

The implementation process is as follows:

1. Assemble a dataset according to the constraint and objective.
2. After the dataloader reads the data, data indices are rearranged based on DP load balancing among the encoders.
3. Data is rearranged using all_to_all communication based on the index positions.
4. The encoder performs load-balanced computation.
5. All-to-all communication is performed according to the original indices to redistribute the embedded data for LLM load balancing.
6. The LLM performs load-balanced computation.

### How to Enable

Add the parameter `--use-data-balance` to the training launch script to enable online data load balancing.

```shell
GPT_ARGS="
    ...
    --use-data-balance \
"
```

Note: Currently, only ViT load balancing is supported.
