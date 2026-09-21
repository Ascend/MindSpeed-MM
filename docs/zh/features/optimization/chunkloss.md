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

## 忽略 token 跳过（SFT 稀疏场景）

在 SFT 等场景下，`shift_labels` 中常有大量 token 被置为 `ignore_index`（prompt 段、padding 等），它们对 loss 与梯度的贡献**恒为 0**，但默认仍会随其余 token 一起被投影到 `lm_head`、白白消耗算力与激活显存。ChunkLoss 会在**进入 `lm_head` 投影之前**把这些 token 丢弃，从而按被 mask 比例节省投影计算与激活显存（mask 越多、收益越大）。两条计算路径均支持：

- ChunkLoss（`chunk_loss`）：拼回完整 `shift_labels`、过滤有效 token 后重新分块再计算；
- CCE（`chunk_loss_cce_fused`）：在进入 vocab-tile 流式 kernel 前过滤掉 `ignore_index` 行。

**该优化对训练无任何数值影响、无需额外开关**：满足条件时自动生效，其 loss 与梯度和不跳过的稠密路径逐元素等价（仅存在 fp32/bf16 求和顺序级别的浮点噪声）。

**生效条件**（同时满足）：

- 计算方式为 `default` 或 `per_token_loss`——这两种都是「所有有效 token 的 CE 求和 ÷ 一个标量 alpha（全量有效 token 数）」，总 loss 与是否丢弃被 mask 的 token 无关，故可安全丢弃；
- 序列中**部分**（而非全部、也非零个）token 被 mask；
- **任意 batch size 均可**——标量-sum 归约与 batch 结构无关，多样本 batch 的有效 token 会被展平成单条稠密序列后重新分块，结果仍与稠密路径逐元素一致。

**回退到稠密路径的情形**（此时行为与未开启该优化完全一致，仅无额外收益）：

- 计算方式为 `per_sample_loss`——它对**每个样本各除以自己的 alpha**（该样本有效 token 数 × batch）后再相加，一个 token 的贡献取决于它属于哪个样本；而丢弃 mask token 需要把整个 batch 展平成一条序列，会丢掉样本边界，无法再对回各自的 alpha，因此不做此优化（对应 `reduction != "sum"` 或 `alpha` 非标量）；
- 序列中没有任何 token 被 mask，或全部被 mask。

## 注意事项

### 已知约束

- `enable_chunk_loss` 与 `enable_dynamic_chunk_loss` 请勿混用：框架不做冲突校验，若同时开启，模型侧按静态分块使能，而loss计算的分块大小由动态分块参数推导，二者行为不一致。
