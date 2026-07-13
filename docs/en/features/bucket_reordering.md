# Data Load Balancing (Data Bucketing and Reordering)

## Data Bucketing Training

Bucketing and reordering data can achieve better load balancing at the data layer.

There are two approaches to data load balancing:

1. Data bucketing: Performance takes priority. Configure `priority_mode` as `data_bucketing_img`. If not configured, data bucketing is used by default.

2. Data reordering: Accuracy takes priority. Configure `priority_mode` as `data_reordering_img`.

## How to Use (Qwen2VL)

### Bucketing for Qwen2VL

In `examples/qwen2vl/data_2b.json`, modify `sampler_type` under `dataloader_param` to `BucketBatchSampler`, and configure `priority_mode` as `data_reordering_img`, as shown below:

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
