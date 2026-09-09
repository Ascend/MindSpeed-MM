# ChunkLoss

## 适用后端

FSDP2 + MCORE

## 背景与挑战

在训练多模态理解模型时，`lm_head` 的输出维度（即词表大小 `vocab_size`）通常远大于模型的隐空间维度 `hidden_size`。传统损失计算方式需要在中间显式构造一个形状为 `[bs, seq, vocab_size]` 的logits张量，这会带来显著的显存峰值，且词表越大或序列越长，该峰值越明显。此外，在动态shape场景下，这一操作还容易引发大块内存碎片，进一步加剧显存管理的负担。

## 解决方案

通过对序列维度进行分块（chunking），将loss计算拆分为多个长度为`sub_seq`的子段依次进行。在完成每个子段的前向计算后，立即执行对应的反向传播，从而避免同时保留整个序列的logits。这样一来，任意时刻最多只需缓存长度为 `sub_seq` 的logits，显著降低了显存峰值。

## 使用方法

ChunkLoss支持FSDP2系与mcore两类后端。FSDP2系有两种配置方式：原生FSDP2（native FSDP2，推荐）与基于Megatron的FSDP2（megatron-FSDP2，过渡态，将退出）；mcore后端目前由qwen3_5模型提供，通过模型config中的 `use_chunk_loss`/`chunk_size` 字段开启（示例见 `examples/mcore/qwen3_5/`），不在本文展开。ChunkLoss不改变loss的计算方式，可与默认、按样本粒度（per sample loss）、按token粒度（per token loss）配合使用，这三种计算方式的说明详见 [VLM模型loss计算方式](./vlm_model_loss_calculate_type.md)。

### 原生FSDP2（推荐）

在模型YAML配置文件的 `features` 段开启ChunkLoss：

```yaml
features:
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
```

### 参数说明

| 配置项 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `enable_chunk_loss` | bool | `false` | 开启静态分块ChunkLoss，按固定块大小切分；块大小由 `chunkloss_plan.chunk_size` 指定。 |
| `enable_dynamic_chunk_loss` | bool | `false` | 开启动态分块ChunkLoss，按总量自适应分块；总量由 `chunkloss_plan.total_chunk_size` 指定。 |
| `chunkloss_plan.apply_module` | str | `lm_head` | 应用ChunkLoss的模块（要求该模块为 `nn.Linear`，否则报错）。 |
| `chunkloss_plan.chunk_size` | int | `1024` | 静态分块时每块的大小（token数），仅 `enable_chunk_loss` 生效。 |
| `chunkloss_plan.total_chunk_size` | int | `4096` | 动态分块时单次计算的总token上限，仅 `enable_dynamic_chunk_loss` 生效，每块大小按批大小自动推导。 |

可参考 `examples/qwen3_5/qwen3_5_4B_config.yaml`。

### 基于Megatron的FSDP2（过渡态，将退出）

> 基于Megatron的FSDP2为过渡方案，后续将逐步退出，新增模型请优先使用原生FSDP2。

在支持ChunkLoss的理解模型配置文件 `model.json` 中，通过 `loss_cfg` 字段进行设置，示例如下：

```json
"loss_cfg": {
    "compute_mode": "chunk",
    "chunk_size": 1024
}
```

- `compute_mode`：
  - 设为 `"default"` 表示使用原始的loss计算方式；
  - 设为 `"chunk"` 则启用ChunkLoss静态分块功能，按固定长度对序列分块后计算loss；
  - 设为 `"dynamic_chunk"` 则启用ChunkLoss动态分块功能，自适应调整分块大小。
- `chunk_size`：
  - 当`compute_mode`设为`"chunk"`时：表示指定序列分块后，每个子序列的最大长度（即每个chunk所包含的token数量）；
  - 当`compute_mode`设为`"dynamic_chunk"`时：表示"每个子序列长度 × 批次大小（batch_size）"的最大长度（用于约束动态分块的总计算量，避免显存溢出）。

通过合理配置 `chunk_size`，可在保证训练正确性的同时有效控制显存占用。

## 注意事项

### 已知约束

- `enable_chunk_loss` 与 `enable_dynamic_chunk_loss` 请勿混用：框架不做冲突校验，若同时开启，模型侧按静态分块使能，而loss计算的分块大小由动态分块参数推导，二者行为不一致。
