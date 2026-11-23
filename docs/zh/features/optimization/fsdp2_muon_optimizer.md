# FSDP2 Muon优化器

## 适用后端

FSDP2 + MCORE

## 背景与挑战

Muon（Momentum Orthogonalized by Newton-Schulz）是一类面向矩阵参数的优化器。它对二维权重矩阵的更新方向进行正交化处理，可以作为AdamW之外的优化选择。

Muon的主要优势在于能够利用神经网络隐藏层权重的矩阵结构，对momentum更新方向做正交化约束，使二维权重矩阵的更新方向更接近良条件的谱范数更新。在部分公开实验中，Muon表现出更好的样本效率和计算效率，即用更少训练时间或FLOPs达到相近loss；但实际收益仍依赖模型结构、batch size、学习率和训练阶段，需要结合业务任务验证。

公开使用案例包括：

- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) 在1T MoE规模上使用Muon/MuonClip训练；
- [DeepSeek-V4 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) 介绍了 DeepSeek-V4 训练使用的 Muon 算法。
- [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/README_CN.md) 使用Muon优化器训练，并建议继续训练或LoRA微调时使用Muon；
- [NVIDIA NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) 给出了在Qwen3-235B-A22B SFT和Qwen2.5-7B DAPO场景中使用Muon的示例。

本仓库实现部分参考 [MoonshotAI/Moonlight示例版Muon](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py) 、 [KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py)，并为FSDP2后端增加了DTensor分片支持。优化器实现在 `mindspeed_mm/optimizer/muon/`，参数分组逻辑在 `mindspeed_mm/fsdp/optimizer/muon_routing.py`。

## 解决方案

在FSDP2后端，Muon优化器先将参数组拆分为Muon参数和AdamW回退参数：

- 除下列情况外，二维权重使用Muon。
- 以下参数使用AdamW回退：
  - 偏置（名称以 `.bias` 结尾）；
  - `nn.Embedding` 模块以及名称为 `lm_head` 或 `output_layer` 的模块的参数；
  - 名称包含 `embedding`、`embed_tokens`、`output_layer`、`lm_head` 或 `muon_fallback_param_keywords` 中任一关键词的参数（忽略大小写）；
  - 至少有一个维度大小为 1 的二维权重（对向量做正交化只会将其归一化）；
  - 非二维参数（`muon_per_expert` 选中的权重除外，见下文）。
- 拆分后的各参数组保留学习率、权重衰减、免衰减分组和 `lr_scaling_plan` 等设置。可设置 `no_decay_params` 或 `no_decay_modules`，以排除指定参数或模块类型的权重衰减。两个列表默认均为空。

Muon更新流程如下：

1. 用梯度的指数移动平均值（EMA）更新动量缓冲区，默认使用 Nesterov 动量。
2. 用 Newton-Schulz（NS）迭代正交化更新方向：
   - 对长边超过短边 1.5 倍、迭代 5 或 10 步且在本地计算的矩阵，使用 FP16 Gram NS。该方法在较小的 Gram 矩阵 X·Xᵀ 上迭代，并在指定步数重新从 X 计算 Gram 矩阵以控制舍入误差。
   - 其他情况下使用 BF16 标准 NS。
   - 归一化范数使用 FP32 计算。
   - 更新在跨 rank 交换等过程中保持 BF16 或 FP16 计算精度，因为将计算结果转换为更高精度不会增加有效精度。向 FP32 参数写回更新时使用 FP32 加法。
   - 在 NPU 上，`torch_catlass` 提供适用于 Ascend 950PR&950DT系列产品的内核时，符合形状条件的矩阵使用 CATLASS SYRK。其余形状使用 PyTorch。CATLASS 内核缺失或不受支持时，优化器在 NPU 上发出警告。
3. 应用权重衰减，再根据 `lr`、`matched_adamw_rms` 和矩阵形状缩放并写回更新。

### 分片参数

在FSDP2中，参数以DTensor形式分片。Muon根据Newton-Schulz成本估算，为每个符合条件的行分片二维权重分配一个负责计算的rank（owner）。Owner通过all-to-all收集完整矩阵，完成正交化，再将各rank对应的行发回。成本估算考虑标准NS、Gram NS、Gram重算、CATLASS内核选择及调用开销。矩阵分批交换。不同批次的通信与计算可按工作负载和后端能力重叠。交换前，每个参与的进程组执行一次all-reduce，检查各rank的参数标识、形状、梯度是否存在、调度设置是否一致，并验证本地行布局。此检查要求各rank的参数组和网格成员一致。

逐头和逐专家更新可以按各自的头块或专家块执行独立的owner交换。符合分布式Gram NS条件的逐头列分片保持分片状态。各rank计算FP32局部Gram矩阵，并以FP32求和。不受支持的布局回退到DTensor重分布。

整矩阵 owner 交换支持以下环境变量：

- `MINDSPEED_MM_MUON_OWNER_WAVE_BYTES`：单个owner在一批中收集的完整矩阵的最大字节数。默认值为1 GiB。超过该值的单个矩阵独占一批。
- `MINDSPEED_MM_MUON_OWNER_MAX_INFLIGHT`：分别限制并行收集窗口和待发回结果的数量。两项默认值均为2。
- `MINDSPEED_MM_MUON_OWNER_SCATTER_COMM`：当owner进程组覆盖所有rank时，设为 `1` 可使用独立进程组发回结果。默认值为 `0`。

批次预算只统计收集的输入矩阵，优化器总内存还涉及其他开销。交换缓冲区、保留的更新、堆叠后的owner批次及Newton-Schulz工作区也会占用内存。增加同时处理的批次数可能增加内存占用，而优化器耗时未必下降。评估通信重叠和owner负载时，请在相同设置下记录收集、NS和发回各阶段耗时、未被重叠隐藏的等待时间、优化器总耗时、峰值内存及各rank上owner的计算时间。

### 分片布局与执行路径

`Shard(d)` 中的 `d` 表示**张量**维度，设备网格维度指定共享此分片的rank。逐专家权重的张量形状为 `[experts, rows, columns]`。

| 参数布局 | 执行路径 | 配置说明 |
| --- | --- | --- |
| 整个二维矩阵，一个普通 `Shard(0)`，其余网格维度复制 | 各rank的本地梯度行与预期行区间一致时，使用owner all-to-all | 使用连续行分片，网格维度名称不限。 |
| 逐头，一个普通 `Shard(0)`，各分片边界均落在头之间 | 对完整头执行本地 NS | FSDP 分片数整除头数即可满足条件。只要边界没有切分任何头，也接受不均匀或空分片。 |
| 逐头，一个普通 `Shard(1)`，头形状和迭代步数符合条件 | 分布式 Gram NS，以 FP32 汇总局部 Gram 矩阵 | 单头需满足 `columns > 1.5 * head_rows`，迭代步数为 5 或 10。 |
| 逐头，一个普通 `Shard(0)` 或 `Shard(1)` 的其余受支持情况 | 逐头owner all-to-all | 行分片可以切分头，不符合分布式Gram条件的列分片也走此路径。 |
| 逐专家，在名为 `ep` 的网格维度上可选 `Shard(0)`，在另一网格维度上有一个普通 `Shard(1)` | 在行分片组内执行逐专家owner all-to-all | EP保持不同专家独立，eFSDP切分每个专家矩阵的行。此布局仍需为行分片专家执行通信。 |
| 其他布局，包括多个参与切分的分片维度和未支持的 eHSDP 布局 | 回退到 all-gather 或重分布 | 快速路径不接受 `Partial` placement 或 `Shard` 子类。回退要求 DTensor 支持目标重分布。 |

逐专家实现通过网格维度名称 `ep` 识别专家分片。改名或省略名称可能导致专家维也被收集。回退时，已识别的 EP 专家维保持分片，矩阵维复制。单 rank 的逐头分片在本地计算。

### 逐头和逐专家 Muon

模型模块通过 `mindspeed_mm/fsdp/optimizer/muon_metadata.py` 中的 `register_muon_layout` 声明布局。将参数登记在直接持有它的模块上。启用 `muon_per_head` 或 `muon_per_expert` 后，FSDP2优化器读取这些声明。每个启用的模式至少需要一个符合条件的可训练参数的布局声明。其余符合条件的矩阵继续按整矩阵使用Muon。

Kimi K3模块在构建时声明以下更新块，Kimi K3.1 Pro复用相应模块时也使用这些更新块。

- Delta Attention 的 Q、K、V 为独立投影，每个投影沿行方向切分为等大的连续头块。
- MLA 按头切分查询投影（`q_proj` 或 `q_b_proj`）及 `kv_b_proj`。`kv_b_proj` 中每个头的 K 和 V 行作为一个块处理。
- 路由专家的 `gate_up_proj` 和 `down_proj` 均按专家分别更新。每个专家的 gate 和 up 仍合并在 `gate_up_proj` 的同一个矩阵中。

### QK-Clip

当观察到的logit超过阈值时，QK-Clip会重新缩放注意力权重。设置 `qk_clip_threshold` 后，Kimi MLA在训练中记录每个头在softmax前的最大logit。优化器更新后，对最大值超过阈值的头，令 `gamma = qk_clip_threshold / max_logit`，并执行以下缩放：

- 查询内容权重乘以 `gamma ** qk_clip_alpha`；
- 键内容权重乘以 `gamma ** (1 - qk_clip_alpha)`；
- 查询的旋转位置编码权重乘以 `gamma`。

在Ulysses或KV all-gather上下文并行下，各rank将本地负责的头的指标记录在包含所有头的向量中。随后在查询权重的设备网格内对各rank的指标求最大值。QK-Clip需要 `optimizer: muon`、`muon_per_head` 和负责绑定查询及键权重并提供指标的模型钩子。Kimi MLA的eager路径和NPU Flash Attention路径提供这些指标。Kimi K3.1 Pro的DSA Ascend C稀疏注意力路径缺少这些指标。

在Ulysses中，本地头占据该向量的连续区间，其余头填 `-inf`。归约前排除打包序列中的填充。查询权重的设备网格必须覆盖参与计算查询token和各头的所有rank，才能得到每个头的全局最大值。代码已实现指标放置和归约，各种Ulysses、ring、打包序列和激活重计算配置仍需进行端到端模型验证。即使关闭裁剪，也要保留注意力反向传播所需的最大值和求和辅助数据。

### 更新语义与数值差异

| 变化 | 影响 |
| --- | --- |
| 在相同系数、步数和分块设置下由 owner 收集、计算并发回 | 改变 NS 的执行位置。数据交换保留张量值，批处理内核的舍入仍可能不同。 |
| Gram 改写、对称半矩阵乘积和融合多项式 | 在精确算术下等价，FP16/BF16 和运算顺序会引入有限精度差异。 |
| 系数序列或迭代步数 | 改变有限步更新算法。默认仍使用Polar Express，早期实现使用单组Keller系数。可显式选择 `ns_coefficients: keller` 来对比旧系数。 |
| 逐头或逐专家正交化 | 改变独立求解的矩阵块，因此也改变更新方向。 |
| QK-Clip | 根据训练时的 logit 在优化器更新后缩放 Q/K 权重。指标或配置校验失败时，权重和已保存的优化器状态保持原值。 |

动量缓冲区由运行累积和改为 EMA。梯度和动量系数相同时，精确算术下 EMA 等于原累积和乘以 `1 - momentum`。这个正比例缩放在加入归一化的 epsilon 之前保持方向。epsilon 和浮点舍入会影响最终更新。验证长期训练精度时，应在相同系数、步数、布局和内核设置下比较损失。

### 数值规则与验证范围

| 规则 | 依据与验证范围 |
| --- | --- |
| Gram多项式改写与对称乘积 | 在 `X X^T` 上迭代可将同一Gram矩阵的多个多项式因子组合后再应用到 `X`。这些因子在精确算术下可交换。有限精度下仍需进行数值检查。 |
| `long_side > 1.5 * short_side` | 这是根据矩形与方阵乘积成本选取的分派规则。实际性能分界随硬件而异。 |
| Gram重算和5/10步限制 | 从 `[2, 5, 7]` 中选择早于最后一步的重算点。5步时在第2步后重算。10步时在第2、5、7步后重算。普通GEMM与保持对称性的SYRK需要分别校准。 |
| CATLASS行数阈值32/256/512、批大小阈值32和列数阈值8192/16384 | 成本模型可估计何时只计算一个三角区域更快，具体阈值则依据单个CATLASS构建的基准测试，其他构建可能需要不同的值。多项式路径还要求行数能被32整除且Gram张量连续。 |

阈值测试应报告准确的代码提交、NPU型号、CANN、PyTorch、`torch_npu` 和 `torch_catlass` 版本，以及形状、数据类型、系数、迭代步数、预热次数、重复次数和计时方法。应在目标部署构建上复测，再判断这些阈值是否合适。历史校准和分析结果不足以证明新源码或训练基线下的长期训练精度。

十步路径中的乘积方向：Newton-Schulz使用两类乘积。`gram_product` 计算矩形矩阵的 `X @ X^T`，其中转置是Gram矩阵定义的一部分。`symmetric_product` 计算两个在精确算术下对称的方阵的乘积。该乘积使用 `A @ B^T` 与 `A @ B` 在精确算术下等价，但转置会放大反对称舍入误差 `A @ (B^T - B)`，该误差会在迭代中累积。

PyTorch内核使用 `A @ B` 计算对称乘积。CATLASS SYRK使用 `A @ B^T`，其多项式路径通过镜像下三角保持操作数严格对称。两条路径计算的是同一精确乘积的不同有限精度形式。

### Megatron 适配

Megatron优化器适配器为普通参数组构造同一个 `Muon` 类。因此可以复用系数序列、本地NS和CATLASS辅助函数。FSDP2的owner交换及逐头、逐专家路由依赖DTensor布局和FSDP模型构建流程，Megatron TP分片需要单独适配。

适配时需确定全局矩阵形状和分片方向，提供全局归一化和进程组，复现分块路由并映射优化器及检查点状态。分布式 Gram 要求每个本地列切片包含完整的行集合。归一化使用各 rank 局部平方范数的 FP32 总和，每段开始时也以 FP32 汇总局部 Gram 矩阵。QK-Clip 还需要收集跨头和 token 的全局注意力最大值，并绑定相应的查询及键权重。TP 和 CP 组合需单独验证。

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
  ns_coefficients: polar_express
```

Kimi K3可以启用逐头、逐专家Muon和QK-Clip：

```yaml
training:
  optimizer: muon
  muon_per_head: true
  muon_per_expert: true
  qk_clip_threshold: 100.0
  qk_clip_alpha: 0.5
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `training.optimizer` | str | `adamw` | 优化器类型，取值 `adamw`、`adamw_swap` 或 `muon`。 |
| `training.matched_adamw_rms` | float | `0.2` | 控制Muon更新量级与AdamW更新RMS的匹配程度。 |
| `training.muon_momentum` | float | `0.95` | Muon内部SGD momentum的动量系数。 |
| `training.ns_steps` | int | `5` | Newton-Schulz正交化迭代步数；步数越大正交化计算越充分，但开销也会增加。 |
| `training.muon_fallback_param_keywords` | list[str] | `null` | 在内置规则之外，追加触发 AdamW 回退的参数名关键词。 |
| `training.lr` | float | `5e-5` | 基础学习率；Muon参数会在基础学习率上结合 `matched_adamw_rms` 和矩阵形状做更新幅度调整。 |
| `training.weight_decay` | float | `0.0` | 权重衰减系数。 |

#### 其他FSDP2参数

- **`ns_coefficients`**
  - 说明：Newton-Schulz系数序列，每次迭代使用一组 `(a, b, c)`。迭代步数超过序列长度时沿用最后一组。
  - 可选值：
    - `polar_express`（默认）：[Polar Express论文](https://arxiv.org/abs/2505.16932)中的五步序列。
    - `keller`：Keller Jordan的原始单组系数 `(3.4445, -4.7750, 2.0315)`。
    - `you`：Jiacheng You的[五步序列](https://x.com/YouJiacheng/status/1905861218138804534)。

- **`muon_per_head`**
  - 说明：根据模型声明的逐头布局，独立正交化每个头的权重块。
  - 默认值：`false`。

- **`muon_per_expert`**
  - 说明：根据模型声明的逐专家布局，独立正交化每个专家矩阵。
  - 默认值：`false`。

- **`qk_clip_threshold`**
  - 说明：每头最大注意力logit的有限非负阈值。需要 `muon_per_head` 和兼容的指标提供者。
  - 默认值：`null`（关闭）。

- **`qk_clip_alpha`**
  - 说明：取值范围为 `[0, 1]`，用于在查询（`gamma ** alpha`）和键（`gamma ** (1 - alpha)`）之间分配QK-Clip缩放量。
  - 默认值：`0.5`。

### mcore后端

通过 `--optimizer muon` 启用，且需在model.json的 `patch` 段设置 `"muon_optimizer": true`。Mcore Muon不能与 `--use-distributed-optimizer` 同时使用。上述FSDP2参数路由规则仅适用于FSDP2。

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `muon_optimizer`（model.json `patch` 段） | bool | `false` | mcore侧启用Muon的接线开关。 |
| `--matched-adamw-rms` | float | `0.2` | 含义同FSDP2侧 `matched_adamw_rms`。 |
| `--muon-momentum` | float | `0.95` | 含义同FSDP2侧 `muon_momentum`。 |
| `--ns-steps` | int | `5` | 含义同FSDP2侧 `ns_steps`。 |

#### 示例脚本

以 `examples/wan2.2/5B/t2v/pretrain.sh` 为起点。设置Muon优化器参数，并在 `examples/wan2.2/5B/t2v/pretrain_model.json` 中启用 `muon_optimizer`。

## 注意事项

1. 在FSDP2中，符合条件的二维矩阵使用Muon。逐专家布局声明也会选中三维专家权重。Muon优化器中的其他参数组使用AdamW回退。
2. 如上所述，FSDP2在各rank之间交换分片以计算正交化更新。Owner批次预算仅限制收集的输入大小，峰值内存还包括其他开销。
3. 可根据训练稳定性和性能要求调整 `ns_steps`。快速验证可使用较小的步数，正式训练建议结合损失曲线和吞吐量评估步数的影响。
4. `matched_adamw_rms` 会影响Muon更新幅度。修改学习率时，建议同时观察该参数对收敛的影响。
5. 加载受支持的旧Muon检查点时，使用保存的动量系数将 `muon_buffer` 乘以 `1 - momentum`，转换为 `momentum_buffer`。如果两个键都存在，以当前缓冲区为准。FSDP2分布式检查点通过Muon钩子读取旧字段，并补齐已知的新参数组字段的默认值。严格加载仍要求学习率等必要字段。CPU测试覆盖普通Muon和扁平化MultiOptimizer的检查点保存后加载及下一次优化器更新。多rank EP训练续跑仍需验证。
6. QK-Clip在更新参数前校验指标和配置。指标缺失或无效时可能需要重新执行前向计算。有效指标会在校验阈值和alpha之前缓存，因此修正相应配置后，重试时可能无需再次执行前向计算。学习率为零的参数组会跳过裁剪，并保留已记录的指标供下一次更新使用。分布式训练的各rank必须使用相同的裁剪设置。
7. 在NPU Flash Attention中，QK-Clip使用内核未接收的 `[batch, seq]` 填充掩码，在计算每头最大值时排除填充的查询行。这适用于右侧填充的批次、尾部填充的打包批次（包括分块打包注意力）以及缓存查询。受支持的填充路径只传入因果掩码，不传入键的填充掩码，因此启用QK-Clip的Kimi训练拒绝左侧填充和内部填充。打包统计值按各序列的头顺序归约。使用ring或本地查询KV all-gather上下文并行时，带填充的批次会报错。方形填充掩码可用于指标计算，但现有内核掩码分派仍不支持 `batch == seq`。
