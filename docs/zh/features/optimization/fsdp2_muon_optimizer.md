# FSDP2 Muon优化器

## 适用后端

FSDP2 + MCORE

## 背景与挑战

Muon（Momentum Orthogonalized by Newton-Schulz）是一类面向矩阵参数的优化器。它对二维权重矩阵的更新方向进行正交化处理，可以作为AdamW之外的优化选择。

Muon的主要优势在于能够利用神经网络隐藏层权重的矩阵结构，对momentum更新方向做正交化约束，使二维权重矩阵的更新方向更接近良条件的谱范数更新。在部分公开实验中，Muon表现出更好的样本效率和计算效率，即用更少训练时间或FLOPs达到相近loss；但实际收益仍依赖模型结构、batch size、学习率和训练阶段，需要结合业务任务验证。

公开使用案例包括：

- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) 在1T MoE规模上使用Muon/MuonClip训练；
- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/README_CN.md) 使用Muon优化器训练，并建议继续训练或LoRA微调时使用Muon；
- [NVIDIA NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) 给出了在Qwen3-235B-A22B SFT和Qwen2.5-7B DAPO场景中使用Muon的示例。

本仓库实现部分参考 [MoonshotAI/Moonlight示例版Muon](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) 、 [KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py)；并在其实现思路上增加了新FSDP2后端的DTensor分片聚合与重新分片适配。

## 解决方案

FSDP2后端下的Muon优化器会先根据参数名称和形状拆分参数组：

- 二维矩阵参数，且参数名不以 `.bias` 结尾、不命中Muon回退规则，使用Muon更新；
- 名称包含 `embedding`、`embed_tokens`、`output_layer`、`lm_head` 的参数，会使用AdamW回退；
- 其他参数自动回退到AdamW更新逻辑；
- 原有的学习率、权重衰减、no decay分组等配置继续保留。

Muon更新流程如下：

1. 对Muon参数使用SGD momentum累积梯度方向；
2. 将更新方向转换为bfloat16，并通过Newton-Schulz迭代做近似正交化；
3. 对权重执行weight decay，并应用正交化后的更新。

在FSDP2场景下，参数可能是DTensor分片。Muon在计算正交化更新前，会将分片参数的更新方向聚合为replicate形态；计算完成后，再按原始DTensor placements重新分片，保证优化器更新和FSDP2参数布局保持一致。

## 使用方法

### FSDP2后端

在FSDP2 YAML配置的 `training` 段将 `optimizer` 设置为 `muon` 即可启用，其余Muon参数（`matched_adamw_rms`、`muon_momentum`、`ns_steps`）按需追加：

```yaml
training:
  lr: 1.0e-5
  weight_decay: 0
  optimizer: muon        # 默认为 adamw
  matched_adamw_rms: 0.2
  muon_momentum: 0.95
  ns_steps: 5
  # 可选：在内置回退规则外，追加业务模型中需要回退 AdamW 的参数名关键词。
  # 例如 custom_head 会让 custom_head.weight 使用 AdamW fallback。
  muon_fallback_param_keywords:
    - custom_head
    - router
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `training.optimizer` | str | `adamw` | 优化器类型，取值 `adamw`、`adamw_swap` 或 `muon`。 |
| `training.matched_adamw_rms` | float | `0.2` | 控制Muon更新量级与AdamW更新RMS的匹配程度。 |
| `training.muon_momentum` | float | `0.95` | Muon内部SGD momentum的动量系数。 |
| `training.ns_steps` | int | `5` | Newton-Schulz正交化迭代步数；步数越大正交化计算越充分，但开销也会增加。 |
| `training.muon_fallback_param_keywords` | list[str] | 内置 `embedding`、`embed_tokens`、`output_layer`、`lm_head` | 追加需要使用AdamW回退逻辑的参数名关键词，配置后在内置规则基础上追加；Qwen/HuggingFace风格的 `embed_tokens`、`lm_head` 默认回退，不需要为Qwen3单独配置。 |
| `training.lr` | float | `5e-5` | 基础学习率；Muon参数会在基础学习率上结合 `matched_adamw_rms` 和矩阵形状做更新幅度调整。 |
| `training.weight_decay` | float | `0.0` | 权重衰减系数。 |

### mcore后端

通过 `--optimizer muon` 启用，且需在model.json的 `patch` 段设置 `"muon_optimizer": true`（否则megatron原生优化器工厂会报 `muon optimizer is not supported.`）。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `muon_optimizer`（model.json `patch` 段） | bool | `false` | mcore侧启用Muon的接线开关。 |
| `--matched-adamw-rms` | float | `0.2` | 含义同FSDP2侧 `matched_adamw_rms`。 |
| `--muon-momentum` | float | `0.95` | 含义同FSDP2侧 `muon_momentum`。 |
| `--ns-steps` | int | `5` | 含义同FSDP2侧 `ns_steps`。 |

#### 示例脚本

`examples/wan2.2/5B/t2v/pretrain.sh`（配置见 `examples/wan2.2/5B/t2v/pretrain_model.json` 的 `muon_optimizer` 字段，默认 `false`）。

## 注意事项

### 已知约束

1. Muon只会作用于满足条件的二维矩阵参数，其余参数会自动使用AdamW回退逻辑，不需要手动拆分参数。
2. FSDP2分片参数会在Muon正交化计算前临时聚合，计算后重新分片；该过程会带来额外通信和计算开销。
3. `ns_steps` 可根据训练稳定性和性能需求调整。短跑通可使用较小值，正式训练建议结合loss曲线和吞吐表现验证。
4. `matched_adamw_rms` 会影响Muon更新量级，修改学习率时建议同步观察该参数对收敛的影响。
5. mcore侧不能与 `--use-distributed-optimizer` 同时使用（框架有校验，同时开启会报错）。
