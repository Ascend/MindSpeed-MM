# 数据负载均衡(数据分桶重排序)

## 问题分析

在多模态模型训练中，不同样本的图片/视频 token 数量差异较大，导致：

- 同一 micro-batch 内的样本长度差异大，padding 浪费严重
- 不同 DP rank 之间的计算负载不均衡，出现快慢卡等待
- 训练吞吐量受限于最慢的 DP rank

上述问题进一步引发以下连锁效应：

- **梯度聚合通信效率低下**：由于各 DP rank 的计算量不均衡，先完成计算的 rank 需要等待最慢的 rank，导致梯度 AllReduce 通信阶段存在大量空闲等待时间，通信-计算重叠率低
- **显存利用率低**：为对齐同一 micro-batch 内最长的样本，短样本需要大量 padding，这些 padding token 占用显存但不产生有效梯度，造成显存浪费
- **训练步间耗时波动大**：不同 batch 的样本长度分布不一致，导致各步训练耗时差异大，整体吞吐不稳定

数据分桶重排序通过对数据进行智能分组和排序，使同一 batch 内的样本长度更加接近，同时使不同 DP rank 之间的计算量更加均衡，从而提升训练效率。

## 解决方案

数据负载均衡的方案分为两种：

1. **数据分桶（data_bucketing_img）**：性能优先，按图片 token 数对数据进行分桶，同一桶内的数据组成 batch，减少 padding 浪费。若不配置 `priority_mode`，默认为数据分桶。

2. **数据重排（data_reordering_img）**：精度优先，在分桶的基础上对数据进行重排序，使得训练数据的分布更加均匀，避免因数据顺序导致的训练偏差。

| 方案 | priority_mode 配置 | 优先级 | 特点 |
|------|-------------------|--------|------|
| 数据分桶 | `data_bucketing_img`（默认） | 性能 | 减少 padding，提升训练吞吐量 |
| 数据重排 | `data_reordering_img` | 精度 | 在分桶基础上保证数据分布均匀性 |

## 使用方法

### Qwen2VL 的数据分桶使用方法

在 `examples/qwen2vl/data_2b.json` 中，修改 `dataloader_param` 下的 `sampler_type` 为 `BucketBatchSampler`，并配置 `priority_mode`：

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

### 配置参数说明

| 参数 | 说明 | 取值 |
|------|------|------|
| `sampler_type` | 采样器类型，启用分桶需设置为 `BucketBatchSampler` | `BucketBatchSampler` |
| `priority_mode` | 负载均衡策略 | `data_bucketing_img`（默认，性能优先）/ `data_reordering_img`（精度优先） |
| `drop_last` | 是否丢弃最后一个不完整的 batch | `true` / `false` |
| `data_sharding` | 是否对数据进行分片（分布式训练时建议开启） | `true` / `false` |
| `shuffle` | 是否在每个 epoch 开始时打乱数据顺序 | `true` / `false` |

### 最佳实践

1. **性能优先场景**：使用默认的 `data_bucketing_img`，最大化减少 padding，提升训练吞吐量
2. **精度敏感场景**：使用 `data_reordering_img`，在保证负载均衡的同时维持数据分布的均匀性
3. **长视频训练**：对于视频生成模型，分桶效果尤为显著，可大幅减少 padding 浪费
4. **混合分辨率数据**：当训练数据中包含不同分辨率的图片/视频时，强烈建议启用分桶

### 注意事项

- 当前已支持 Qwen2VL 模型，其他模型的支持正在扩展中
- 启用分桶后，数据的原始顺序会被改变，但不影响训练的收敛性
- `data_reordering_img` 相比 `data_bucketing_img` 会有少量额外计算开销
- 在 DP=1 的单卡场景下，分桶主要减少 padding；在 DP>1 时，还能改善负载均衡
