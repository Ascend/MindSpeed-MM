# ChunkLoss

## 背景与挑战

在训练多模态理解模型时，`lm_head` 的输出维度（即词表大小 `vocab_size`）通常远大于模型的隐空间维度 `hidden_size`。传统损失计算方式需要在中间显式构造一个形状为 `[bs, seq, vocab_size]` 的 logits 张量，这会带来显著的显存峰值，且词表越大或序列越长，该峰值越明显。此外，在动态 shape 场景下，这一操作还容易引发大块内存碎片，进一步加剧显存管理的负担。

## 解决方案

通过对序列维度进行分块（chunking），将 loss 计算拆分为多个长度为`sub_seq`的子段依次进行。在完成每个子段的前向计算后，立即执行对应的反向传播，从而避免同时保留整个序列的 logits。这样一来，任意时刻最多只需缓存长度为 `sub_seq` 的 logits，显著降低了显存峰值。

## 使用方法

当前MindSpeed MM支持的理解模型loss计算公式详见[文档](vlm_model_loss_calculate_type.md)，当前chunkloss功能已支持其中的默认方式、按样本粒度计算（per sample loss）以及按token粒度计算（per token loss）

在每个支持chunkloss的理解模型配置文件model.json中，可通过 loss_cfg 字段进行相关设置，示例如下：

```json
"loss_cfg": {
    "compute_mode": "default",
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

针在多模态理解模型中启用 ChunkLoss 特性后，通过合理设置 `chunk_size`，可在显著降低显存峰值的同时保持相同的损失曲线。
