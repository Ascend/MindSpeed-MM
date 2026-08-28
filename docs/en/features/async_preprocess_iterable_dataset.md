# AsyncPreprocessIterableDataset

## Overview

`AsyncPreprocessIterableDataset` is an asynchronous preprocessing wrapper for streaming data scenarios. It decouples the preprocessing of individual samples from the main training thread and dispatches it to multiple background workers for concurrent execution, while preserving the logical order of samples.

This capability is currently provided in two training pipelines in this repository:

- The Megatron implementation is located at `mindspeed_mm/data/datasets/qwen2vl_dataset.py`
- The FSDP2 implementation is located at `mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py`

It mainly serves the Hugging Face `streaming=True` `IterableDataset` scenario, and is suitable for tasks where multimodal samples such as text, images, videos, and audio require relatively heavy CPU preprocessing before training, including template concatenation, tokenization, and modality input organization.

The core objectives of this feature include:

1. Reduce the time the training main loop spends waiting for data preprocessing.
2. Maintain sample order consistency across data parallel replicas.
3. Reuse existing preprocessing logic without rewriting the current `preprocess_fn` signature.

The related parameters are described as follows:

- `async_preprocess`: Whether to enable streaming asynchronous preprocessing. This parameter mainly takes effect when `streaming: true` is set. When enabled, the dataset is wrapped as `AsyncPreprocessIterableDataset` after DP (data-parallel) sharding, and samples are read and asynchronously preprocessed on the fly during training. When disabled, the existing `dataset.map(..., batched=True)` preprocessing path is used instead.
- `async_preprocess_buffer_size`: The buffer depth of asynchronous preprocessing, used to approximately control the scale of raw sample tasks that can be simultaneously in flight in the internal task queue and result queue. In the current implementation, this parameter takes effect per raw sample rather than per batch: the producer submits one sample task at a time, and workers also invoke preprocessing at the granularity of a single sample. Therefore, it is better understood as "how many sample tasks to prefetch/buffer". This parameter determines the degree of decoupling among the producer, workers, and the main thread, trading off throughput against memory usage. When not explicitly set, it is normalized together with `preprocessing_num_workers`: if neither is configured, the default value starts from `8`; if only `preprocessing_num_workers` is configured, `buffer_size` uses the same value; if only `async_preprocess_buffer_size` is configured, the number of workers is automatically filled to a reasonable value not exceeding the number of CPUs.
- Pay attention to the difference and relationship between `async_preprocess_buffer_size` and `preprocessing_batch_size`: `preprocessing_batch_size` controls the number of samples fed into `preprocess_fn` at a time under the `dataset.map(..., batched=True)` path, while `async_preprocess_buffer_size` controls how many samples "waiting to be processed or waiting to be output in order" can be buffered under the asynchronous path. The two operate at different levels and do not refer to the same "batch" concept. In the current implementation, the asynchronous path temporarily wraps a single sample into a batch of length 1 before calling `preprocess_fn`. Therefore, when `async_preprocess` is enabled, `preprocessing_batch_size` is not used to control the processing granularity of workers, and `async_preprocess_buffer_size` does not change how many samples are actually processed in a single preprocessing call.

## Motivation and Background

In non-streaming mode, training data can usually be preprocessed once via `map` and then consumed by the DataLoader. In streaming mode, however, the data itself is a continuously produced iterable object, and the training process must read and process it on the fly. This gives rise to several typical problems:

1. Per-sample preprocessing is expensive. Multimodal samples often involve steps such as template construction, image/video information organization, audio feature preparation, and tokenizer encoding. If all of these are executed serially in the main training thread, throughput is directly reduced.
2. The upstream streaming data source is not suitable for direct concurrent consumption by multiple threads. If multiple workers iterate over the same `IterableDataset` simultaneously, it is easy to introduce duplicate reads, out-of-order sequences, or inconsistent sample offsets across different ranks.
3. Streaming training requires stable ordering. Even when concurrency is introduced, the sample order observed on the training side must remain consistent with the upstream logical order; otherwise, resumable training, alignment validation, and multi-replica consistency will all be affected.
4. Streaming data is also constrained by the data loading mode. The current repository supports two modes, `base` and `sampler`. Reading data in the dataset's own iteration order, the `base` mode is compatible with `IterableDataset`, but does not support shuffle. The `sampler` mode relies on `len(dataset)` to generate global indices and samples via `dataset[idx]`, without support for `IterableDataset`, and is therefore not applicable to streaming reads.

Given the above constraints, `AsyncPreprocessIterableDataset` does not have multiple workers concurrently consume the upstream stream. Instead, it adopts a design of "single-threaded sequential reading of upstream data + multi-worker concurrent preprocessing + in-order rearrangement of outputs". The current asynchronous preprocessing capability is implemented based on the `base` mode; the `sampler` mode and the global shuffle that depends on its indexing mechanism are not applicable to streaming scenarios. If out-of-order capability is needed later, a more appropriate approach is to perform local shuffle within a window or buffer.

## Design Principles

The overall processing pipeline of `AsyncPreprocessIterableDataset` is as follows:

```text
load_dataset(..., streaming=True)
    -> align_dataset(...)
    -> DistributedIterableDataset
    -> AsyncPreprocessIterableDataset
    -> DataLoader / StatefulDataLoader
    -> DataCollator
    -> Model forward and training loop
```

The key design points are as follows:

1. Distributed sharding is performed first, followed by asynchronous preprocessing.
   - Under the Megatron path, the training dataset is first wrapped as `DistributedIterableDataset` inside `get_qwen2vl_dataset()`, and then wrapped as `AsyncPreprocessIterableDataset` as needed.
   - Under the FSDP2 path, the processing order remains consistent with Megatron: DP sharding is performed before asynchronous preprocessing.
   - `DistributedIterableDataset` is responsible for sharding the raw sample stream by DP rank and must be executed before `AsyncPreprocessIterableDataset`; the latter only performs asynchronous preprocessing on a single sub-stream and outputs samples in order. If the order were reversed, with asynchronous preprocessing performed before sharding, each rank would first process the entire upstream data stream and then discard most of the results, which not only causes duplicate preprocessing but also introduces additional CPU, thread, and queue overhead. In addition, post-sharding operates on the preprocessed output stream rather than the original sample stream. Once sample filtering or one-to-many expansion occurs during preprocessing, the sharding boundaries between ranks may drift, thereby disrupting stable alignment across replicas.

2. The upstream data streams are always consumed sequentially by a single producer thread.
   - The producer is responsible for assigning a globally increasing `sequence_idx` to each sample, which serves as the sole basis for subsequent reordering.

3. The preprocessing task is executed concurrently by multiple background workers.
   - Workers do not directly access the upstream dataset; they only process samples that have already entered the task queue, thereby avoiding multi-thread contention on the upstream iterator.

4. In the result output stage, reordering is performed based on `sequence_idx`.
   - Even if different workers complete in different orders, the main iterator ultimately yields results to the training side in the original order.

5. The configuration layer controls this capability through `async_preprocess` and `async_preprocess_buffer_size`.
   - The Megatron configuration is defined in `mindspeed_mm/data/data_utils/func_utils/convert.py`
   - The FSDP2 configuration is defined in `mindspeed_mm/fsdp/data/data_utils/func_utils/convert.py`

In the current implementation, if the training data has `streaming: true` and `async_preprocess: true` enabled, the dataset construction process enters this asynchronous path.

## Detailed Explanation of Core Mechanisms

### 1. Configuration Normalization

During class initialization, `buffer_size` and `num_workers` are normalized first:

- When neither is configured, `buffer_size` is set to `8`, and `num_workers` set to `min(buffer_size, cpu_count)` by default.
- When only `buffer_size` is configured, `num_workers` automatically takes a reasonable value not exceeding the number of CPUs.
- When only `num_workers` is configured, `buffer_size` is set to the same value.

The purpose of this set of rules is to balance throughput and memory usage by default, avoiding worker starvation caused by an overly shallow queue or extra cache pressure caused by an overly deep queue.

From a tuning perspective, when `buffer_size` is too small, the task queue and result queue are more likely to be drained or filled up, workers are prone to starvation, and the main thread is more likely to wait directly for data, which manifests as reduced training throughput. When `buffer_size` is too large, more unconsumed samples and their preprocessing results are cached, increasing CPU memory usage, which is especially noticeable for large-sample scenarios such as images, videos, and audio. If the training side is inherently slower than the preprocessing side, further increasing `buffer_size` often yields limited benefit.

In practice, the ratio of the average per-sample preprocessing time to the average per-sample consumption time on the training side can be used as a reference. If preprocessing is significantly slower than training, increase `preprocessing_num_workers` first, and then set `buffer_size` to a value no smaller than `num_workers` to avoid worker starvation. If the two are comparable, `buffer_size` can be tuned to the range from `num_workers` to `2 * num_workers` to absorb fluctuations in per-sample processing time. If training is significantly slower than preprocessing, there is usually no need to set `buffer_size` too large; keeping it around `num_workers` is sufficient.

### 2. Single-Sample Batching Adaptation

The `preprocess_fn` in the repository takes a batch dictionary as input. Therefore, `AsyncPreprocessIterableDataset` does not pass a single sample directly to it; instead, it first performs a lightweight wrapping:

1. Wrap each field of a single sample into a list of length 1.
2. Call the existing `preprocess_fn(batch_dict)`.
3. Unpack the returned batched result back into a single-sample list.

This approach ensures that streaming asynchronous preprocessing can reuse existing preprocessors, without the need to additionally implement a single-sample version for the streaming scenario.

### 3. Producer for Sequential Reading

Inside `__iter__()`, a producer thread is started to sequentially traverse the upstream dataset. The producer thread is responsible for only two tasks:

1. Generate an incrementing `sequence_idx` for each sample.
2. Put `(sequence_idx, item)` into `task_queue`.

This ensures that the upstream streaming data source always has a single read entry point, fundamentally avoiding the ordering uncertainty caused by concurrent reads.

### 4. Worker for Concurrent Preprocessing

Multiple worker threads take tasks from `task_queue` and execute `_preprocess_item()`. After each worker finishes processing, it writes the result to `result_queue` in the form of `(message_type, payload, extra)`.

The current implementation mainly has three types of messages:

1. `result` indicates that preprocessing of the sample corresponding to a certain `sequence_idx` is complete.
2. `done` indicates that a worker has processed all tasks and exited.
3. `error` indicates that an exception was thrown inside the producer or worker, and the entire iteration chain needs to be terminated.

### 5. In-order Reordering for Output Stability

Because different samples take different amounts of time to preprocess, the order in which workers return results is usually unstable. However, "order consistency" here does not mean that different ranks see exactly the same samples; rather, it means that each rank strictly outputs results in the order of its own deterministic substream. Differences in worker processing speed only affect the waiting time, not the output order. The code guarantees this mainly through the following layers of mechanisms:

1. Sharding occurs before asynchronous preprocessing.
   - `DistributedIterableDataset` has already performed DP sharding on the outer layer according to the original sample indices. The Megatron path samples by `idx % num_dp == dp_rank`, and the FSDP2 path splits samples to each rank using the same modulo rule.
   - When entering `AsyncPreprocessIterableDataset`, each rank is already facing only its own sample substream. No matter how fast or slow the subsequent workers are, they cannot change this sharding result, nor affect the sample boundaries of other ranks.

2. Each rank has only one producer thread that assigns sequential IDs.
   - The producer thread sequentially iterates over the current rank's substream through `enumerate(self.dataset)` and assigns a monotonically increasing `sequence_idx` to each sample.
   - This `sequence_idx` is determined only by the iteration order of the upstream substream, not by the completion timing of workers, so it is the only reliable ordering basis within the current rank.

3. Workers can complete out of order, but cannot output out of order.
   - After a worker finishes processing a task, it only writes `("result", sequence_idx, processed_items)` into `result_queue`.
   - After the main iterator receives a result, it first places it into `pending_results[sequence_idx]`, where `pending_results` is used to cache results that are "already complete but cannot yet be output", and `next_sequence_idx` indicates the next output sequence number that is permitted to be yielded.
   - Only when `next_sequence_idx` already exists in `pending_results` does the main iterator actually yield the corresponding result and increment `next_sequence_idx` by one. In other words, faster workers can only "arrive early and be cached," but they cannot "output early."

4. Derived results from a single raw sample are output as a whole in order.
   - `_preprocess_item()` may return a `processed_items` list, representing a set of training samples generated from a single raw sample after preprocessing.
   - Once the main iterator hits a certain `sequence_idx`, it first outputs this group of `processed_items` in full, and then advances to the next `sequence_idx`. Therefore, even if one-to-many expansion occurs during the preprocessing stage, the derived results of different original samples will not interleave with each other.

The following uses a two-rank example for illustration, assuming the original sample stream is `s0, s1, s2, s3, s4, s5` with a DP size is 2:

- After outer-layer sharding, rank 0 sees only `s0, s2, s4`, which are renumbered within this rank as `sequence_idx = 0, 1, 2`.
- After outer-layer sharding, rank 1 sees only `s1, s3, s5`, which are likewise renumbered within this rank as `sequence_idx = 0, 1, 2`.

If, on rank 0, the worker processing `s2` is faster than the worker processing `s0`, then `result_queue` may receive `sequence_idx = 1` before `sequence_idx = 0`. In this case, the main iterator first places the result with index 1 into `pending_results`, but does not output it immediately; only after index 0 arrives does it produce results in the order `0 -> 1 -> 2`. Rank 1 follows exactly the same rule. Therefore, what is actually stabilized across different ranks is not the "completion order", but rather "post-sharding substream order + the `sequence_idx` order". Differences in worker speed only change how long a result waits in the buffer, and do not change the final output order on each rank.

### 6. Distributed Consistency

`AsyncPreprocessIterableDataset` does not perform DP sharding itself; it relies on the outer `DistributedIterableDataset` to first split the data for each DP replica.

- On the Megatron side, `mpu.get_data_parallel_world_size()` and `mpu.get_data_parallel_rank()` are used to determine the subsequence that the current rank should consume.
- On the FSDP2 side, sharding ownership within the DP group is determined based on the parallel state and the current rank.

The benefit of this approach is that each DP replica performs asynchronous preprocessing only on its own sample sub-stream, which avoids multiple replicas seeing the same samples and also avoids cross-replica ordering inconsistency.

If the order is reversed, the basis for DP sharding would change from the original sample sequence index to the post-preprocessing output index. Once filtering or expansion of samples occurs during the preprocessing stage, the stable alignment across replicas would be broken.

### 7. Exception Propagation and Resource Reclamation

The implementation uses `stop_event`, a task-end sentinel, and a unified error message to ensure fast exit when an exception occurs.

1. If the producer encounters an error while iterating over the upstream data, it writes the error into `result_queue`.
2. If a worker encounters an error while performing preprocessing, it immediately sends an error message and triggers `stop_event`.
3. After receiving the error message, the main thread aborts the iteration and re-raises the exception back to the training side.
4. The `finally` branch uniformly reclaims the background threads to avoid dangling threads.

## Usage Examples

### 1. Megatron Example: Qwen2.5Omni

Under the Megatron path, the training entry script for Qwen2.5Omni is `examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`, the training program entry is `pretrain_vlm.py`, and the default data configuration file is `examples/qwen2.5omni/data_7b.json`.

To enable `AsyncPreprocessIterableDataset`, it is recommended to add at least the following fields to the data configuration:

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-Omni-7B",
            "use_fast_tokenizer": true,
            "split_special_tokens": false,
            "image_max_pixels": 262144,
            "image_min_pixels": 0,
            "video_max_pixels": 16384,
            "video_min_pixels": 0,
            "video_fps": 2.0,
            "video_maxlen": 128
        },
        "basic_parameters": {
            "template": "qwen2_omni",
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            "train_on_prompt": false,
            "mask_history": false,
            "preprocessing_batch_size": 1000,
            "preprocessing_num_workers": 16,
            "max_samples": null,
            "tool_format": null,
            "streaming": true,
            "async_preprocess": true,
            "async_preprocess_buffer_size": 16
        },
        "attr": {
            "system": null,
            "images": null,
            "videos": "videos",
            "audios": "audios",
            "messages": "messages",
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": null,
            "function_tag": null,
            "system_tag": null
        }
    },
    "dataloader_param": {
        "dataloader_mode": "base",
        "drop_last": true,
        "sampler_type": "BaseRandomBatchSampler",
        "collate_param": {
            "model_name": "qwen2vl",
            "ignore_pad_token_for_loss": true
        },
        "pin_memory": true,
        "shuffle": false
    }
}
```

The following shows an example of how to launch it:

```bash
bash examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh
```

### 2. FSDP2 Example: Qwen3Omni

Under the FSDP2 path, the training entry script for Qwen3Omni is `examples/qwen3omni/finetune_qwen3omni_v1.sh`, the training program entry is `mindspeed_mm/fsdp/train/trainer.py`, and the default data configuration file is `examples/qwen3omni/qwen3omni_config_v1.yaml`.

To enable `AsyncPreprocessIterableDataset`, modify `examples/qwen3omni/qwen3omni_config_v1.yaml` as follows:

```yaml
# Parallel strategy
parallel:
  tensor_parallel_size: 1
  fully_shard_parallel_size: auto
  fsdp_plan:
    apply_modules:
      - audio_tower.positional_embedding
      - audio_tower.layers.{*}
      - visual.blocks.{*}
      - visual.merger
      - visual.merger_list.{*}
      - visual
      - model.embed_tokens
      - model.layers.{*}
      - model
      - lm_head
    param_dtype: bf16
    reduce_dtype: fp32
    num_to_forward_prefetch: 1
    num_to_backward_prefetch: 1
  ep_plan:
    apply_modules:
      - model.layers.{*}.mlp.experts
  recompute: true
  recompute_plan:
    apply_modules:
      - model.layers.{*}
  ring_attention_size: 1
  ulysses_parallel_size: 1
  expert_parallel_size: 1

# Data-related configuration
data:
  dataset_param:
    dataset_type: huggingface
    # Dataset attributes
    attr:
      audios: audios
      # images: images
      videos: videos
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant

    # Data preprocessing
    preprocess_parameters:
      model_name_or_path: &HF_MODEL_LOAD_PATH ./ckpt/hf_path/Qwen3-Omni-30B-A3B-Instruct
      use_fast_tokenizer: true
      split_special_tokens: false
      use_audio_in_video: true
      image_max_pixels: 262144
      image_min_pixels: 1024
      video_max_pixels: 16384
      video_min_pixels: 256
      video_fps: 2.0
      video_maxlen: 128
      audio_sampling_rate: 16000

    basic_parameters:
      cutoff_len: 262144
      template: qwen3_omni
      enable_thinking: false
      train_on_prompt: false
      mask_history: false
      dataset_dir: ./data
      dataset: &DATASET_PATH ./data/mllm_format_llava_instruct_data.json
      cache_dir: ./data/cache_dir
      overwrite_cache: false
      preprocessing_batch_size: 128
      preprocessing_num_workers: 32
      max_samples: null
      streaming: true
      async_preprocess: true
      async_preprocess_buffer_size: 32

  # Data loading
  dataloader_param:
    pin_memory: true
    shuffle: false
    dataloader_mode: base
    drop_last: true
    sampler_type: BaseRandomBatchSampler
    num_workers: 16
    collate_param:
      model_name: qwen3omni
      ignore_pad_token_for_loss: true

# Model configuration
model:
  model_id: qwen3_omni_moe
  model_name_or_path: *HF_MODEL_LOAD_PATH
  trust_remote_code: true
  attn_implementation: flash_attention_2
  freeze:
    - visual.patch_embed
    - visual.blocks
    - visual.merger_list
    - visual.pos_embed
    - visual.merger
    - audio_tower
  loss_cfg:
    loss_type: default   # If you want raw loss in model, loss_type can be set to "raw".
    router_aux_loss_coef: 0.0
  enable_chunk_loss: true  # If loss_type is set to "raw", enable_chunk_loss must be set to false.
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
  use_grouped_expert_matmul: true

# Training configuration
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  seed: 42
  lr: 1.0e-5
  lr_decay_style: cosine
  lr_warmup_ratio: 0.1
  weight_decay: 0
  train_iters: 5000
  clip_grad: 0.0
  init_model_with_meta_device: false
  optimizer: adamw
  adam_fused: true
  save_interval: 10000
  load: ./ckpt/mm_path/Qwen3-Omni-30B-A3B-Instruct
  save: ./save_path
  use_deter_comp: false
  plugin:
    - mindspeed_mm/fsdp/models/qwen3omni
    - mindspeed_mm/fsdp/data/datasets/huggingface
  no_load_optim: true
  no_load_rng: true
  no_save_optim: true
  no_save_rng: true

# Tool configuration
tools:
  profile:
    enable: false
    profile_type: static
    ranks: [0]
    static_param:
      level: level1
      with_stack: false
      with_memory: false
      record_shapes: false
      with_cpu: true
      save_path: ./profiling
      start_step: 10
      end_step: 11
      data_simplification: false
      aic_metrics_type: PipeUtilization
  memory_profile:
    enable: false
    start_step: 1
    end_step: 2
    save_path: ./memory_snapshot
    dump_ranks: [0]
    stacks: all
    max_entries: null
    mem_info: false
```

The following shows an example of how to launch it:

```bash
bash examples/qwen3omni/finetune_qwen3omni_v1.sh
```

### 3. Suggestions

1. It is recommended to set `preprocessing_num_workers` to `4` or `8`, and tune it gradually based on CPU resources and sample complexity.
2. `async_preprocess_buffer_size` is generally recommended to be no smaller than `preprocessing_num_workers` to reduce queue blocking.
3. If reproducibility is the primary concern, it is recommended to fix the random seed when enabling this feature, and to first disable other randomization configurations that may change the sample order.
4. If the current configuration still uses a batch sampler that depends on `len(dataset)`, switch to a loading method that does not depend on the dataset length before enabling streaming asynchronous preprocessing.
   