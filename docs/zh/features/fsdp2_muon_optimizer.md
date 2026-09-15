# FSDP2 Muon优化器

Muon（Momentum Orthogonalized by Newton-Schulz）是一类面向矩阵参数的优化器。它对二维权重矩阵的更新方向进行正交化处理，可以作为 AdamW 之外的优化选择。

Muon 的主要优势在于能够利用神经网络隐藏层权重的矩阵结构，对 momentum 更新方向做正交化约束，使二维权重矩阵的更新方向更接近良条件的谱范数更新。在部分公开实验中，Muon 表现出更好的样本效率和计算效率，即用更少训练时间或 FLOPs 达到相近 loss；但实际收益仍依赖模型结构、batch size、学习率和训练阶段，需要结合业务任务验证。

公开使用案例包括：

- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) 在 1T MoE 规模上使用 Muon/MuonClip 训练；
- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/README_CN.md) 使用 Muon 优化器训练，并建议继续训练或 LoRA 微调时使用 Muon；
- [NVIDIA NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) 给出了在 Qwen3-235B-A22B SFT 和 Qwen2.5-7B DAPO 场景中使用 Muon 的示例。

本仓库实现部分参考 [MoonshotAI/Moonlight 示例版 Muon](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) 、 [KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py)；并在其实现思路上增加了新 FSDP2 后端的 DTensor 分片聚合与重新分片适配。

## 更新流程

FSDP2 后端下的 Muon 优化器会先根据参数名称和形状拆分参数组：

- 二维矩阵参数，且参数名不以 `.bias` 结尾、不包含 `embedding`、不包含 `output_layer`，使用 Muon 更新；
- 其他参数自动回退到 AdamW 更新逻辑；
- 原有的学习率、权重衰减、no decay 分组等配置继续保留。

Muon 更新流程如下：

1. 对 Muon 参数使用 SGD momentum 累积梯度方向；
2. 将更新方向转换为 bfloat16，并通过 Newton-Schulz 迭代做近似正交化；
3. 对权重执行 weight decay，并应用正交化后的更新。

在 FSDP2 场景下，参数可能是 DTensor 分片。Muon 在计算正交化更新前，会将分片参数的更新方向聚合为 replicate 形态；计算完成后，再按原始 DTensor placements 重新分片，保证优化器更新和 FSDP2 参数布局保持一致。

## 使用方法

在 FSDP2 YAML 配置中，将 `training.optimizer` 设置为 `muon` 即可启用。

```yaml
training:
  lr: 1.0e-5
  weight_decay: 0
  optimizer: muon
  matched_adamw_rms: 0.2
  muon_momentum: 0.95
  ns_steps: 5
```

## 参数详解

- **`optimizer`**
  - 描述：选择优化器类型。
  - 取值：`adamw` 或 `muon`。

- **`matched_adamw_rms`**
  - 描述：控制 Muon 更新量级与 AdamW 更新 RMS 的匹配程度。
  - 默认值：`0.2`。

- **`muon_momentum`**
  - 描述：Muon 内部 SGD momentum 的动量系数。
  - 默认值：`0.95`。

- **`ns_steps`**
  - 描述：Newton-Schulz 正交化迭代步数。
  - 默认值：`5`。
  - 说明：步数越大，正交化计算越充分，但开销也会增加。

- **`lr`**
  - 描述：基础学习率。
  - 说明：Muon 参数会在基础学习率上结合 `matched_adamw_rms` 和矩阵形状做更新幅度调整。

- **`weight_decay`**
  - 描述：权重衰减系数。

## 注意事项

1. Muon 只会作用于满足条件的二维矩阵参数，其余参数会自动使用 AdamW 回退逻辑，不需要手动拆分参数。
2. FSDP2 分片参数会在 Muon 正交化计算前临时聚合，计算后重新分片；该过程会带来额外通信和计算开销。
3. `ns_steps` 可根据训练稳定性和性能需求调整。短跑通可使用较小值，正式训练建议结合 loss 曲线和吞吐表现验证。
4. `matched_adamw_rms` 会影响 Muon 更新量级，修改学习率时建议同步观察该参数对收敛的影响。
