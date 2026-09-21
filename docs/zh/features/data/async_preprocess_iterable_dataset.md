# AsyncPreprocessIterableDataset

## 概述

`AsyncPreprocessIterableDataset` 是面向流式数据场景的异步预处理包装器，用于在保持样本逻辑顺序不变的前提下，将单条样本的预处理工作从训练主线程中解耦出来，交由多个后台 worker 并发执行。

当前仓库已经在两条训练链路中提供了该能力：

- Megatron 路径实现位于 `mindspeed_mm/data/datasets/qwen2vl_dataset.py`
- FSDP2 路径实现位于 `mindspeed_mm/fsdp/data/datasets/huggingface/qwen2vl_dataset.py`

它主要服务于 HuggingFace `streaming=True` 的 `IterableDataset` 场景，适用于文本、图像、视频、音频等多模态样本在训练前需要进行模板拼接、tokenize、模态输入整理等较重 CPU 预处理的任务。

该特性的核心目标包括：

1. 降低训练主循环等待数据预处理的时间。
2. 保持数据并行副本之间的样本顺序一致性。
3. 在不改写现有 `preprocess_fn` 签名的前提下复用已有预处理逻辑。

相关参数说明如下：

- `async_preprocess`：是否启用流式异步预处理。该参数主要在 `streaming: true` 时生效；开启后，数据集会在 DP 分片之后包装为 `AsyncPreprocessIterableDataset`，训练过程中边读取边异步预处理；关闭时，则继续走现有的 `dataset.map(..., batched=True)` 预处理路径。
- `async_preprocess_buffer_size`：异步预处理的缓冲深度，用于近似控制内部任务队列和结果队列中可同时在途的原始样本任务规模。在当前实现里，这个参数按单条原始样本生效，而不是按批次生效：producer 每次提交的是一条样本任务，worker 也是按单样本粒度调用预处理，因此它更适合理解为“预取/缓存多少条样本任务”。该参数决定 producer、worker 和主线程之间的解耦程度，在吞吐和内存占用之间做权衡。未显式设置时，会与 `preprocessing_num_workers` 一起做归一化：二者都未配置时默认起点为 8；只配置 `preprocessing_num_workers` 时，会回落为同样大小；只配置 `async_preprocess_buffer_size` 时，worker 数会自动补齐为不超过 CPU 数量的合理值。
- 特别说明：`async_preprocess_buffer_size` 与 `preprocessing_batch_size` 的区别和关系：`preprocessing_batch_size` 控制的是 `dataset.map(..., batched=True)` 路径下，每次送入 `preprocess_fn` 的样本数；`async_preprocess_buffer_size` 控制的是异步路径下可缓存多少条“等待处理或等待按序输出”的样本。两者作用层级不同，不是同一个“batch”概念。当前实现中，异步路径会把单条样本临时封装成长度为 1 的 batch 再调用 `preprocess_fn`，因此开启 `async_preprocess` 后，`preprocessing_batch_size` 不用于控制 worker 的处理粒度，而 `async_preprocess_buffer_size` 也不会改变单次预处理实际处理多少条样本。

## 动机与背景

在非流式模式下，训练数据通常可以先通过 `map` 一次性完成预处理，再交给 DataLoader 消费；但在流式模式下，数据本身是一个持续产出的可迭代对象，训练过程需要边读边处理，此时会遇到几个典型问题：

1. 单条样本预处理开销大。多模态样本往往包含模板构造、图像/视频信息整理、音频特征准备和 tokenizer 编码等步骤，如果全部放在训练主线程里串行执行，会直接拉低吞吐。
2. 上游流式数据源不适合被多个线程同时直接消费。若多个 worker 同时遍历同一个 `IterableDataset`，很容易引入重复读取、顺序错乱或不同 rank 样本偏移不一致的问题。
3. 流式训练要求顺序稳定。即使引入并发，也必须保证训练侧看到的样本顺序与上游逻辑顺序一致，否则断点续训、对齐验证和多副本一致性都会受到影响。
4. 流式数据还受到数据加载模式的限制。当前仓库支持 `base` 和 `sampler` 两种模式：`base` 模式按 dataset 自身的迭代顺序读取，兼容 `IterableDataset`，但不支持 shuffle；`sampler` 模式依赖 `len(dataset)` 生成全局索引，并通过 `dataset[idx]` 取样，不支持 `IterableDataset`，因此不适用于流式读取。

基于以上约束，`AsyncPreprocessIterableDataset` 并不是让多个 worker 并发消费上游流，而是采用“单线程顺序读取上游数据 + 多 worker 并发预处理 + 按序重排输出”的设计。当前异步预处理能力基于 `base` 模式实现；`sampler` 模式以及依赖其索引机制的全局 shuffle 不适用于流式场景，后续若需要引入乱序能力，更合适的方向是在窗口或缓冲区内做局部 shuffle。

## 设计方案

`AsyncPreprocessIterableDataset` 的整体处理链路如下：

```text
load_dataset(..., streaming=True)
    -> align_dataset(...)
    -> DistributedIterableDataset
    -> AsyncPreprocessIterableDataset
    -> DataLoader / StatefulDataLoader
    -> DataCollator
    -> 模型前向与训练循环
```

其中的关键设计点如下：

1. 先做分布式分片，再做异步预处理。
   - Megatron 路径中，训练集在 `get_qwen2vl_dataset()` 内先包装为 `DistributedIterableDataset`，再按需包装为 `AsyncPreprocessIterableDataset`。
   - FSDP2 路径中，处理顺序与 Megatron 保持一致，先保证 DP 分片，再启动异步预处理。
   - `DistributedIterableDataset` 负责按 DP rank 切分原始样本流，必须先于 `AsyncPreprocessIterableDataset` 执行；后者只负责对单个子流做异步预处理并按序输出。如果反过来先做异步预处理再做分片，每个 rank 都会先处理整份上游流，再丢弃大部分结果，不仅会造成重复预处理，还会带来额外的 CPU、线程和队列开销。此外，后置分片面对的是预处理后的输出流，而不是原始样本流；一旦预处理阶段发生样本过滤或一对多展开，rank 之间的分片边界就可能漂移，进而破坏副本间的稳定对齐。

2. 上游数据流始终只由一个 producer 线程顺序消费。
   - producer 负责给每条样本分配全局递增的 `sequence_idx`，这是后续重排的唯一依据。

3. 预处理工作由多个后台 worker 并发执行。
   - worker 不直接访问上游 dataset，只处理已经进入任务队列的样本，从而避免多线程竞争上游迭代器。

4. 结果输出阶段按 `sequence_idx` 做重排。
   - 即使不同 worker 的完成顺序不同，主迭代器最终仍按原始顺序向训练侧产出结果。

5. 配置层通过 `async_preprocess` 与 `async_preprocess_buffer_size` 控制该能力。
   - Megatron 配置定义位于 `mindspeed_mm/data/data_utils/func_utils/convert.py`
   - FSDP2 配置定义位于 `mindspeed_mm/fsdp/data/data_utils/func_utils/convert.py`

在当前实现下，如果训练数据开启了 `streaming: true` 并同时设置 `async_preprocess: true`，数据集构建流程就会进入这一异步路径。

## 核心机制详解

### 1. 配置归一化

类初始化时会先归一化 `buffer_size` 与 `num_workers`：

- 当二者都未配置时，默认 `buffer_size=8`，`num_workers` 取 `min(buffer_size, cpu_count)`。
- 当只配置了 `buffer_size` 时，`num_workers` 自动取不超过 CPU 数量的合理值。
- 当只配置了 `num_workers` 时，`buffer_size` 会回落为同样大小。

这套规则的目的是在默认情况下平衡吞吐与内存占用，避免队列过浅导致 worker 饥饿，或队列过深造成额外缓存压力。

从调参角度看，`buffer_size` 过小时，任务队列和结果队列更容易被取空或堆满，worker 容易饥饿，主线程也更容易直接等待数据，表现为训练吞吐下降；`buffer_size` 过大时，则会缓存更多尚未消费的样本及其预处理结果，增加 CPU 内存占用，对于图像、视频、音频等大样本场景尤为明显。如果训练侧本身慢于预处理侧，继续增大 `buffer_size` 往往收益有限。

实际调参时，可以结合平均单样本预处理耗时与训练侧平均单样本消费耗时的比值来判断：若预处理明显慢于训练，优先增加 `preprocessing_num_workers`，再将 `buffer_size` 设为不小于 `num_workers` 的值以避免 worker 饥饿；若两者接近，可将 `buffer_size` 调到 `num_workers` 到 `2 * num_workers` 区间以吸收样本耗时波动；若训练明显慢于预处理，则通常无需把 `buffer_size` 设得过大，保持在 `num_workers` 附近即可。

### 2. 单条样本批量化适配

仓库中的 `preprocess_fn` 以 batch 字典为输入，因此 `AsyncPreprocessIterableDataset` 不会直接把单条样本传给它，而是会先执行一次轻量封装：

1. 将单条样本的每个字段包装成长度为 1 的列表。
2. 调用已有的 `preprocess_fn(batch_dict)`。
3. 再把返回的 batched 结果拆回单样本列表。

这种做法保证了流式异步预处理可以复用现有预处理器，而不需要为 streaming 场景额外实现一套单样本版本。

### 3. producer 线程负责顺序读取

`__iter__()` 内部会启动一个 producer 线程，专门顺序遍历上游 dataset。producer 线程只负责两件事：

1. 为每条样本生成递增的 `sequence_idx`。
2. 把 `(sequence_idx, item)` 放入 `task_queue`。

这样可以保证上游流式数据源始终只有一个读取入口，从根源上避免并发读取带来的顺序不确定性。

### 4. worker 线程负责并发预处理

多个 worker 线程从 `task_queue` 中取出任务并执行 `_preprocess_item()`。每个 worker 处理完成后，会将结果以 `(message_type, payload, extra)` 的形式写入 `result_queue`。

当前实现里主要存在三类消息：

1. `result`：表示某个 `sequence_idx` 对应的样本预处理完成。
2. `done`：表示某个 worker 已经处理完所有任务并退出。
3. `error`：表示 producer 或 worker 内部抛出了异常，需要终止整个迭代链路。

### 5. 按序重排保证输出稳定性

由于不同样本的预处理耗时不同，worker 返回结果的先后顺序通常是不稳定的。但这里的“顺序一致性”并不是指不同 rank 会看到完全相同的样本，而是指每个 rank 都会严格按照自己那条确定性子流的顺序输出结果；worker 处理速度的差异只会影响等待时间，不会改变输出顺序。代码里主要通过以下几层机制保证这一点：

1. 先分片，再进入异步预处理。
   - `DistributedIterableDataset` 已经在外层按原始样本索引完成 DP 分片。Megatron 路径按 `idx % num_dp == dp_rank` 取样，FSDP2 路径按同样的模运算规则切分到各个 rank。
   - 因此进入 `AsyncPreprocessIterableDataset` 时，每个 rank 面对的已经是只属于自己的样本子流。后续 worker 再快或再慢，都无法改变这个分片结果，也不会影响其他 rank 的样本边界。

2. 每个 rank 只由一个 producer 线程顺序编号。
   - producer 线程通过 `enumerate(self.dataset)` 顺序遍历当前 rank 的子流，并为每条样本分配单调递增的 `sequence_idx`。
   - 这个 `sequence_idx` 只由上游子流的迭代顺序决定，不由 worker 的完成时序决定，所以它是当前 rank 内部唯一可信的顺序基准。

3. worker 只能乱序完成，不能乱序输出。
   - worker 处理完任务后，只会把 `("result", sequence_idx, processed_items)` 写入 `result_queue`。
   - 主迭代器收到结果后，会先放入 `pending_results[sequence_idx]`，其中 `pending_results` 用来缓存“已经完成但还不能输出”的结果，`next_sequence_idx` 表示当前下一个允许对外输出的编号。
   - 只有当 `pending_results` 中已经存在 `next_sequence_idx` 时，主迭代器才会真正 `yield` 对应结果，并将 `next_sequence_idx` 加一。也就是说，较快 worker 的结果最多只能“先到先缓存”，不能“先到先输出”。

4. 单条原始样本的派生结果会作为一个整体按序输出。
   - `_preprocess_item()` 可能返回一个 `processed_items` 列表，表示同一条原始样本经过预处理后生成的一组训练样本。
   - 主迭代器命中某个 `sequence_idx` 后，会先完整输出这一组 `processed_items`，再推进到下一个 `sequence_idx`。因此即使预处理阶段出现一对多展开，不同原始样本的派生结果也不会互相穿插。

可以用一个两 rank 的例子来理解。假设原始样本流是 `s0, s1, s2, s3, s4, s5`，DP size 为 2：

- rank 0 经过外层分片后只看到 `s0, s2, s4`，在本 rank 内部会被重新编号为 `sequence_idx = 0, 1, 2`。
- rank 1 经过外层分片后只看到 `s1, s3, s5`，在本 rank 内部同样会被重新编号为 `sequence_idx = 0, 1, 2`。

如果 rank 0 上处理 `s2` 的 worker 比处理 `s0` 的 worker 更快，那么 `result_queue` 可能先收到 `sequence_idx = 1`，后收到 `sequence_idx = 0`。这时主迭代器会先把编号 1 的结果放进 `pending_results`，但不会立刻输出；只有等编号 0 到达后，才会按 `0 -> 1 -> 2` 的顺序依次产出。rank 1 也完全遵循同样的规则。因此，不同 rank 上真正稳定下来的不是“完成顺序”，而是“分片后的子流顺序 + `sequence_idx` 顺序”；worker 速度差异只会改变结果在缓存里等待多久，不会改变各 rank 的最终输出顺序。

### 6. 分布式一致性

`AsyncPreprocessIterableDataset` 本身不做 DP 分片，它依赖外层的 `DistributedIterableDataset` 先完成每个数据并行副本的数据切分：

- Megatron 侧通过 `mpu.get_data_parallel_world_size()` 与 `mpu.get_data_parallel_rank()` 计算本 rank 应消费的子序列。
- FSDP2 侧通过并行状态和当前 rank 计算数据并行组内的分片归属。

这样做的好处是每个 DP 副本只会对属于自己的样本子流做异步预处理，避免多个副本看到相同样本，也避免跨副本顺序不一致。

如果把顺序反过来，DP 分片依据就会从原始样本序号变成预处理后的输出序号；一旦预处理阶段发生样本过滤或展开，各副本之间的稳定对齐就会被破坏。

### 7. 异常传播与资源回收

实现中使用 `stop_event`、任务结束哨兵和统一的错误消息来保证异常时可以快速退出：

1. 如果 producer 遍历上游数据时出错，会把错误写入 `result_queue`。
2. 如果 worker 在执行预处理时出错，也会立刻发送错误消息并触发 `stop_event`。
3. 主线程收到错误消息后会中止迭代，并将异常重新抛回训练侧。
4. `finally` 分支会统一回收后台线程，避免出现悬挂线程。

## 使用示例

### 1. Megatron 示例：Qwen2.5Omni

Megatron 路径下，Qwen2.5Omni 的训练入口脚本为 `examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh`，训练程序入口为 `pretrain_vlm.py`，默认数据配置文件为 `examples/qwen2.5omni/data_7b.json`。

若要启用 `AsyncPreprocessIterableDataset`，建议在数据配置中至少增加以下字段：

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

启动方式示例如下：

```bash
bash examples/qwen2.5omni/finetune_qwen2_5_omni_7b.sh
```

### 2. FSDP2 示例：Qwen3Omni

FSDP2 路径下，Qwen3Omni 的训练入口脚本为 `examples/qwen3omni/finetune_qwen3omni_v1.sh`，训练程序入口为 `mindspeed_mm/fsdp/train/trainer.py`，默认数据配置文件为 `examples/qwen3omni/qwen3omni_config_v1.yaml`。

若要启用 `AsyncPreprocessIterableDataset`，可在 `examples/qwen3omni/qwen3omni_config_v1.yaml` 中按如下方式修改：

```yaml
# 并行策略
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

# 数据相关配置
data:
  dataset_param:
    dataset_type: huggingface
    # 数据集属性
    attr:
      audios: audios
      # images: images
      videos: videos
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant

    # 数据预处理
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

  # 数据加载
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

# 模型配置
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

# 训练配置
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

# 工具配置
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

启动方式示例如下：

```bash
bash examples/qwen3omni/finetune_qwen3omni_v1.sh
```

### 3. 使用建议

1. `preprocessing_num_workers` 建议从 4 或 8 起步，根据 CPU 资源和样本复杂度逐步调优。
2. `async_preprocess_buffer_size` 通常建议不小于 `preprocessing_num_workers`，以减少队列阻塞。
3. 若优先关注可复现性，建议在开启该特性时同步固定随机种子，并先关闭其他会改变样本顺序的随机化配置。
4. 若当前配置仍使用依赖 `len(dataset)` 的 batch sampler，应先切换到不依赖数据集长度的加载方式，再启用流式异步预处理。
