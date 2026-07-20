# ChunkLoss

## 背景与挑战

在训练多模态理解模型时，`lm_head` 的输出维度（即词表大小 `vocab_size`）通常远大于模型的隐空间维度 `hidden_size`。传统损失计算方式需要在中间显式构造一个形状为 `[bs, seq, vocab_size]` 的 logits 张量，这会带来显著的显存峰值，且词表越大或序列越长，该峰值越明显。此外，在动态 shape 场景下，这一操作还容易引发大块内存碎片，进一步加剧显存管理的负担。

## 解决方案

通过对序列维度进行分块（chunking），将 loss 计算拆分为多个长度为`sub_seq`的子段依次进行。在完成每个子段的前向计算后，立即执行对应的反向传播，从而避免同时保留整个序列的 logits。这样一来，任意时刻最多只需缓存长度为 `sub_seq` 的 logits，显著降低了显存峰值。

## 使用方法

ChunkLoss 当前仅支持 FSDP2 后端，有两种配置方式：原生 FSDP2（native FSDP2，推荐）与基于 Megatron 的 FSDP2（megatron-FSDP2，过渡态，将退出）。ChunkLoss 不改变 loss 的计算方式，可与默认、按样本粒度（per sample loss）、按 token 粒度（per token loss）配合使用，这三种计算方式的说明详见 [VLM 模型 loss 计算方式](vlm_model_loss_calculate_type.md)。

### 原生 FSDP2（推荐）

在模型 YAML 配置文件的 `features` 段开启 ChunkLoss：

```yaml
features:
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
```

`enable_chunk_loss`（静态分块）与 `enable_dynamic_chunk_loss`（动态分块）二选一，分别对应 `chunkloss_plan` 中不同的块大小参数，请勿混用：

- `enable_chunk_loss`：开启静态分块 ChunkLoss，按固定块大小切分，默认 `false`；块大小由 `chunkloss_plan.chunk_size` 指定。
- `enable_dynamic_chunk_loss`：开启动态分块 ChunkLoss，按总量自适应分块，默认 `false`；总量由 `chunkloss_plan.total_chunk_size` 指定。
- `chunkloss_plan`：
  - `apply_module`：应用 ChunkLoss 的模块，默认 `lm_head`。
  - `chunk_size`：静态分块时每块的大小（token 数），默认 `1024`（仅 `enable_chunk_loss` 生效）。
  - `total_chunk_size`：动态分块时单次计算的总 token 上限，默认 `4096`（仅 `enable_dynamic_chunk_loss` 生效，每块大小按批大小自动推导）。

可参考 `examples/qwen3_5/qwen3_5_4B_config.yaml`。

### 基于 Megatron 的 FSDP2（过渡态，将退出）

> 基于 Megatron 的 FSDP2 为过渡方案，后续将逐步退出，新增模型请优先使用原生 FSDP2。

在支持 ChunkLoss 的理解模型配置文件 `model.json` 中，通过 `loss_cfg` 字段进行设置，示例如下：

```json
"loss_cfg": {
    "compute_mode": "chunk",
    "chunk_size": 1024
}
```

- `compute_mode`：
  - 设为 `"default"` 表示使用原始的 loss 计算方式；
  - 设为 `"chunk"` 则启用 ChunkLoss 静态分块功能，按固定长度对序列分块后计算loss；
  - 设为 `"dynamic_chunk"` 则启用 ChunkLoss 动态分块功能, 自适应调整分块大小。
- `chunk_size`：
  - 当`compute_mode`设为`"chunk"`时：表示指定序列分块后，每个子序列的最大长度（即每个 chunk 所包含的 token 数量）；
  - 当`compute_mode`设为`"dynamic_chunk"`：表示"每个子序列长度 × 批次大小（batch_size）"的最大长度（用于约束动态分块的总计算量，避免显存溢出）。

通过合理配置 `chunk_size`，可在保证训练正确性的同时有效控制显存占用。

## 使用效果

在多模态理解模型中启用 ChunkLoss 特性后，通过合理设置 `chunk_size`，可在显著降低显存峰值的同时保持相同的损失曲线。
