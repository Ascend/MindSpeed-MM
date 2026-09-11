# Preprocess On Fly

## Background and Challenges

In non-streaming data loading scenarios, data preprocessing is performed by default via `dataset.map` on the full dataset during the startup phase, with results written to Arrow cache. This approach presents the following bottlenecks in multimodal training scenarios:

- Slow startup: Full preprocessing of large-scale datasets is time-consuming, and the process may time out and exit with an error if data processing takes too long.
- High disk usage: After preprocessing, multimodal data (images and videos) is fully written to disk on a per-sample basis, consuming a large amount of disk space.

To address the challenges above, the Preprocess On Fly strategy has been introduced.

## Solution

- On-demand execution: Attach `preprocess_func` as the dataset's transform via `set_transform`, so that preprocessing is triggered only when each batch is read during training, without writing to disk.
- Parallel prefetching: Leverage the multi-process prefetching mechanism of DataLoader's `num_workers` to mask preprocessing overhead.

## How to Use

**This feature only takes effect for Hugging Face dataset types in non-streaming data loading scenarios**, controlled by the `preprocess_on_fly` parameter. It is currently enabled by default in the Kimi-K2.5 and Qwen3.5 model series. An example of usage is as follows:

```yaml
basic_parameters:
  streaming: false  # Streaming must be disabled
  preprocess_on_fly: true

dataloader_param:
  num_workers: 8  # Mask preprocessing latency through multi-process prefetching
```

### Parameter Details

- `streaming`: the switch for streaming loading. `preprocess_on_fly` takes effect only when `streaming: false` is set. This parameter is disabled by default.
- `preprocess_on_fly`: whether to perform preprocessing during training. The default value is `false`. When enabled, the `set_transform` path is used instead of `dataset.map`, and `preprocessing_batch_size` and `preprocessing_num_workers` do not take effect.
- `num_workers`: the number of DataLoader worker processes, which masks the preprocessing overhead through multi-process prefetching. When `num_workers=0`, preprocessing is executed synchronously in the main process and blocks training.
  