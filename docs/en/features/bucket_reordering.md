# Data Load Balancing (Data Bucketing and Reordering)

## Problem Analysis

In multimodal model training, the number of image/video tokens varies significantly across different samples, leading to:

- Large differences in sample length within the same micro-batch, causing severe padding waste
- Imbalanced computational load across different DP ranks, resulting in fast-slow rank waiting
- Training throughput limited by the slowest DP rank

The above issues further lead to the following chain effects:

- **Inefficient gradient aggregation communication**: Due to the computational imbalance across DP ranks, faster ranks must wait for the slowest one, causing significant idle time during gradient AllReduce communication and reducing communication-computation overlap efficiency.
- **Low memory utilization**: To align with the longest sample in the same micro-batch, short samples require a large amount of padding. These padding tokens occupy memory but do not produce valid gradients, causing memory waste.
- **Large fluctuation in per-step training time**: Different batches have different sample length distributions, causing large variations in per-step training time and unstable overall throughput.

Data bucketing and reordering intelligently groups and sorts data so that sample lengths within the same batch are closer to each other, while also making the computational load more balanced across DP ranks, thereby improving training efficiency.

## Solution

There are two solutions for data load balancing:

1. **Data bucketing (`data_bucketing_img`)**: Performance-first. Groups data into buckets based on image token counts, and batches samples from the same bucket to reduce padding waste. This is the default mode when `priority_mode` is not configured.
2. **Data reordering (`data_reordering_img`)**: Accuracy-first. Reorders data within buckets to ensure a more uniform training data distribution, avoiding training bias caused by data ordering.

| Solution | priority_mode Configuration | Priority | Characteristics |
|------|-------------------|--------|------|
| Data bucketing | `data_bucketing_img` (default) | Performance | Reduces padding and improves training throughput |
| Data reordering | `data_reordering_img` | Accuracy | Ensures uniform data distribution on top of bucketing |

## How to Use

### Data Bucketing for Qwen2VL

In `examples/qwen2vl/data_2b.json`, set `sampler_type` under `dataloader_param` to `BucketBatchSampler`, and configure `priority_mode`:

```json
"dataloader_param": {
    "dataloader_mode": "sampler",
    "drop_last": true,
    "sampler_type": "BucketBatchSampler",
    "priority_mode": "data_reordering_img",
    "collate_param": {
        "model_name": "qwen2vl",
        "ignore_pad_token_for_loss": true
    },
    "pin_memory": true,
    "data_sharding": true,
    "shuffle": true
}
```

### Parameter Descriptions

| Parameter | Description | Value |
|------|------|------|
| `sampler_type` | Sampler type. To enable bucketing, set it to `BucketBatchSampler`. | `BucketBatchSampler` |
| `priority_mode` | Load balancing strategy. | `data_bucketing_img` (default, performance-first)/`data_reordering_img` (accuracy-first) |
| `drop_last` | Whether to drop the last incomplete batch. | `true`/`false` |
| `data_sharding` | Whether to shard the data (recommended to enable during distributed training). | `true`/`false` |
| `shuffle` | Whether to shuffle the data order at the beginning of each epoch. | `true`/`false` |

### Best Practices

1. **Performance-first scenarios**: Use the default `data_bucketing_img` to minimize padding and improve training throughput.
2. **Accuracy-sensitive scenarios**: Use `data_reordering_img` to maintain the uniformity of data distribution while ensuring load balancing.
3. **Long video training**: For video generation models, bucketing is particularly effective and can significantly reduce padding waste.
4. **Mixed-resolution data**: When the training data contains images/videos of different resolutions, it is strongly recommended to enable bucketing.

### Precautions

- Currently supported for Qwen2VL models; support for other models is being expanded.
- After bucketing is enabled, the original order of the data will be changed, but this does not affect training convergence.
- `data_reordering_img` incurs a small amount of additional computational overhead compared with `data_bucketing_img`.
- In a single-card scenario with `DP = 1`, bucketing mainly reduces padding; when `DP > 1`, it also improves load balancing.
