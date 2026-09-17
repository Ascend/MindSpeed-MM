# MagiHuman 使用指南（FSDP2）

## 目录

- [版本说明](#版本说明)
- [前置准备](#前置准备)
- [环境安装](#环境安装)
- [数据集准备及处理](#数据集准备及处理)
- [训练](#训练)
- [已支持特性](#已支持特性)
- [权重布局与转换](#权重布局与转换)
- [环境变量声明](#环境变量声明)

## 版本说明

MagiHuman 是一个 15B 单流音视频扩散 DiT，将 VIDEO、AUDIO 和 TEXT token 打包为一条序列，
联合预测 video/audio 的 flow-matching velocity。主要结构包括：

- 40 层单流 Transformer layer，hidden size 5120；
- GQA attention，40 个 query head、8 个 KV head，head dim 128；
- 首尾各 4 层使用 VIDEO、AUDIO、TEXT 模态独立参数；
- 自定义 `BaseLinear` 和 `NativeMoELinear` 线性层。

本目录提供 MagiHuman text-to-audio-video 的全参数 FSDP2 训练样例。

参考实现：

- 项目源码：[GAIR-NLP/daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman)
- 模型权重：[GAIR/daVinci-MagiHuman](https://huggingface.co/GAIR/daVinci-MagiHuman)
- 迁移使用的上游提交：`209209b7086eba2020c5439265221495a8357322`

2026.07.03：首次支持 MagiHuman text-to-audio-video 微调训练和 FSDP2 后端。

## 前置准备

### 1. 下载模型权重

训练阶段使用 MagiHuman base 分片 safetensors。运行时只接受本地路径，请先下载权重，再在
YAML 中填写本地目录：

```bash
MODEL_ROOT=/path/to/daVinci-MagiHuman
hf download GAIR/daVinci-MagiHuman \
  --include "base/*" \
  --local-dir "${MODEL_ROOT}"
```

下载完成后，权重目录为 `${MODEL_ROOT}/base`。将 YAML 中的
`model.model_name_or_path` 和 `model.checkpoint_path` 指向该目录。

文本编码器、VAE 和 audio encoder 只在离线生成预计算数据时使用，训练进程不会加载这些
编码器。相应模型和处理方式以 MagiHuman 上游仓库为准。

### 2. 准备预计算数据

训练从预计算的 video/audio latent 和 text embedding 开始。数据格式见
[数据集准备及处理](#数据集准备及处理)。

## 环境安装

### 1. 环境准备

请先按 [安装指导](../../docs/zh/pytorch/install_guide.md) 完成 CANN、PyTorch、torch-npu 和
MindSpeed MM 的安装，配套版本以该文档为准。

在 MindSpeed MM 仓库根目录安装当前代码：

```bash
pip install -e .
```

### 2. 准备上游 inference 模块

MagiHuman 的 DiT 使用以 `inference` 为根的绝对导入。训练前需要将固定版本的上游
`inference/` 模块放入模型目录：

```bash
# 以下命令在 MindSpeed MM 仓库根目录执行
SOURCE_ROOT=/path/to/daVinci-MagiHuman-source

git clone https://github.com/GAIR-NLP/daVinci-MagiHuman.git "${SOURCE_ROOT}"
git -C "${SOURCE_ROOT}" checkout 209209b7086eba2020c5439265221495a8357322

cp -r "${SOURCE_ROOT}/inference" \
  mindspeed_mm/fsdp/models/magihuman/inference

sed -i '/dit_model import get_dit/d' \
  mindspeed_mm/fsdp/models/magihuman/inference/model/dit/__init__.py
```

插件注册阶段不会提前导入上游重型依赖，但真正构建模型前必须完成上述操作。训练路径直接构建
`DiTModel`，不要调用上游 `dit_model.py::get_dit`，避免进入 CUDA checkpoint 和 distributed
初始化路径。

## 数据集准备及处理

MagiHuman 数据插件注册名为 `magihuman_precomputed`。dataset 读取干净的 video/audio latent
和 text embedding；`collate_fn` 负责采样 sigma、构造 flow-matching 输入和 target，并完成
单流 packing。

### 1. 合成数据

合成数据只用于验证数据、模型、反向和 optimizer step 是否接通。将 YAML 中的
`magihuman_dataset_custom` 显式改为：

```yaml
magihuman_dataset_custom: &MAGIHUMAN_DATASET_CUSTOM_AV
  with_audio: true
  seed: 42
  synthetic: true
  synthetic_samples: 8
  synthetic_video_grid: [1, 2, 2]
  synthetic_audio_tokens: 2
  synthetic_text_tokens: 3
```

`synthetic` 默认为 `false`。真实训练模式下，如果数据目录或必需子目录不存在，dataset 会直接
报错，不会自动回退到随机数据。

### 2. 真实数据

预计算目录结构如下：

```text
/path/to/dataset/
└── .precomputed/
    ├── latents/*.pt
    ├── audio_latents/*.pt
    └── conditions/*.pt
```

每条样本需要包含以下字段：

| 子目录 | 可用字段名 | 形状 | 说明 |
| --- | --- | --- | --- |
| `latents/` | `video_latent` 或 `latents` | `[Nv, 192]` | patchified video latent |
| `latents/` | `video_grid_thw` 或 `grid_thw` | `(T, H, W)` | 满足 `T×H×W=Nv` |
| `audio_latents/` | `audio_latent` 或 `latents` | `[Na, 64]` | audio latent |
| `conditions/` | `prompt_embeds` 或 `text_embed` | `[Nt, 3584]` | text condition embedding |

三个目录中的文件按相对路径匹配。对于 `latent_<id>.pt`，dataset 也支持
`condition_<id>.pt` 和 `audio_<id>.pt` 的命名方式。

每条样本按 VIDEO、AUDIO、TEXT 顺序组成单流序列；batch 内多条样本沿 token 维拼接。
`collate_fn` 会输出 `cu_seqlens` 和 `max_seqlen`，确保多样本使用 TND 变长注意力时保持样本
边界。只有 video/audio token 参与 flow-matching loss，text token 仅作为条件。

YAML 中的关键数据配置如下：

```yaml
magihuman_dataset_custom: &MAGIHUMAN_DATASET_CUSTOM_AV
  latents_dir: latents
  conditions_dir: conditions
  audio_latents_dir: audio_latents
  with_audio: true
  seed: 42
  timestep_sampling_mode: logit_normal
  timestep_sampling_params:
    loc: 0.0
    scale: 1.0
  synthetic: false

data:
  dataset_param:
    dataset_type: magihuman_precomputed
    preprocess_parameters: {}
    magihuman_dataset_custom: *MAGIHUMAN_DATASET_CUSTOM_AV
    basic_parameters:
      dataset_dir: /path/to/dataset/.precomputed
      dataset: magihuman_precomputed
      max_samples: 10000
  dataloader_param:
    pin_memory: true
    shuffle: false
    dataloader_mode: sampler
    drop_last: true
    sampler_type: BaseRandomBatchSampler
    num_workers: 0
    collate_param:
      model_name: magihuman
      ignore_pad_token_for_loss: false
```

`dataset_dir` 可以指向 `.precomputed`，也可以指向包含该子目录的上一级目录。

## 训练

### 1. 准备工作

运行前检查以下配置：

- `model.model_name_or_path` 和 `model.checkpoint_path` 指向本地 base safetensors 目录；
- `data.dataset_param.basic_parameters.dataset_dir` 指向预计算数据目录；
- `training.plugin` 同时包含模型插件和数据插件：

```yaml
training:
  plugin:
    - mindspeed_mm/fsdp/models/magihuman/magihuman_fsdp2
    - mindspeed_mm/fsdp/data/datasets/magihuman
```

### 2. 配置文件

全参数训练配置为 [magihuman_config_t2av.yaml](magihuman_config_t2av.yaml)，主要字段如下：

```yaml
parallel:
  fully_shard_parallel_size: auto
  fsdp_plan:
    apply_modules:
      - dit.block.layers.{*}
    param_dtype: bf16
    reduce_dtype: fp32
    num_to_forward_prefetch: 2
    num_to_backward_prefetch: 2

features:
  loss_cfg:
    loss_type: raw
  recompute: true
  recompute_plan:
    apply_modules:
      - dit.block.layers.{*}
```

`fully_shard_parallel_size: auto` 根据总 world size 设置 FSDP2 分片组。样例以
`micro_batch_size: 1` 作为保守起点，可根据输入长度和显存情况调整。样例默认只保存模型权重，
需要保存优化器和 RNG 状态时移除 `no_save_optim` 和 `no_save_rng`。

### 3. 启动训练

通过下列脚本来启动训练：

```bash
bash examples/magihuman/finetune_magihuman_t2av.sh
```

卡数按实际环境修改。脚本通过 `torchrun` 启动统一训练入口
`mindspeed_mm/fsdp/train/trainer.py`，YAML 是唯一位置参数。

训练日志能够持续输出 iteration，并且 loss、grad norm 未出现 NaN/Inf，即说明 forward、
backward 和 optimizer step 已接通。首个 iteration 通常包含初始化和编译开销，耗时可能明显
高于后续迭代。启动失败或训练卡住时，可查阅 [FAQ](../../docs/zh/FAQ.md)。

在 8 卡 Ascend 910B3 上按上述配置训练，输入取 video 3840、audio 126、text 640 token
（packed 长度 4606），5000 step 稳态 step time 中位数 4634.8 ms，单卡 71.7 TFLOP/s，峰值显存
51384 MiB/卡。MFU 按单卡峰值 280 TFLOPS 计算为 25.6%，可用
`mindspeed_mm/fsdp/tools/flops_tool/flops_magihuman.py` 按实际硬件复算。

上述结果依赖两项默认开启的 NPU 适配。融合 RoPE 使用 `npu_rotary_mul`：MagiHuman 的 RoPE 宽度
为 96、head dim 为 128，只旋转前 96 维，其余维度保持不变，融合前后前向和梯度均为 bit-exact；
RMSNorm 使用 `npu_rms_norm`，fp32 下最大误差约 1e-6。连续半区 gated MLP 见
[权重布局与转换](#权重布局与转换)。

## 已支持特性

当前样例已验证的特性如下，未列出的框架特性尚未在 MagiHuman 上验证。

| 特性 | 是否支持 | 说明 |
| --- | --- | --- |
| [FSDP2](../../docs/zh/features/fsdp2.md) | 是 | 分片粒度 `dit.block.layers.{*}`，`param_dtype: bf16`、`reduce_dtype: fp32` |
| 全参数微调 | 是 | text-to-audio-video，单流打包序列 |
| 重计算 | 是 | `features.recompute` 配合 `recompute_plan` |
| 序列打包 | 是 | dataset 的 `collate_fn` 沿 token 轴拼接，`cu_seqlens` 驱动 varlen attention |
| NPU 融合算子 | 是 | attention、RoPE、RMSNorm，默认开启 |
| 权重加载与导出 | 是 | DCP 保存与续训、HF 权重在线加载、导出回上游 HF 格式 |
| [Ulysses CP](../../docs/zh/features/unaligned_ulysses_cp.md) | 是 | 开启后 DP 同比减小，需用梯度累积补回 global batch |
| [LoRA 微调](../../docs/zh/features/lora_finetune_fsdp2.md) | 否 | — |
| 张量并行、Ring Attention | 否 | 相关配置项保持为 `1` |
| [Chunk Loss](../../docs/zh/features/chunkloss.md) | 否 | — |
| [Async Activation Offload](../../docs/zh/features/async_activation_offload.md) | 是 | 作用于 `dit.block.layers.{*}`；8 卡 100 步开关对照 loss/grad norm 逐位一致，每步卸载约 838 MiB，吞吐下降约 3% |

## 权重布局与转换

上游 MagiHuman checkpoint 将 gated MLP 的 `up_gate_proj.weight` 按
`[gate_0, linear_0, gate_1, linear_1, ...]` 交错存储。FSDP2 NPU 训练默认在加载后将层
4-39 转为 `[gate..., linear...]` 连续半区布局，从而避免每层执行 stride-2 ViewCopy；层
0-3 使用 GELU7，不参与转换。参数名称、shape、dtype 和 `down_proj` 均不改变。

交错布局下 `swiglu7` 需要用 `x[..., ::2]` 和 `x[..., 1::2]` 取出两个 stride-2 视图。8 卡
profiling 显示，36 个 gated 层每步共产生 72 次 ViewCopy、合计约 1767 ms，占算子耗时的
27.8%，高于两路 MatMul 之和。改为连续半区后 `swiglu7` 使用 `chunk(2, -1)`，两个视图本身
连续，不再触发拷贝；权重只在加载时原地重排，显存不变。层 36-39 的权重按 VIDEO、AUDIO、
TEXT 顺序存放三个连续的模态块，转换按块独立进行。

DCP 是训练内部格式，保存和恢复训练 checkpoint 时无需手动转换。将 `training.load` 指向公共 HF
权重目录并设置 `training.load_format: hf` 时，框架按 `model_id` 选中 MagiHuman 的权重转换流水
线，在 FSDP2 分片前自动完成相同转换。对外导出时必须使用 `MagiHumanConverter` 恢复上游交错
布局：

```bash
mm-convert MagiHumanConverter dcp_to_hf \
  --load_dir /path/to/dcp/iter_0000100 \
  --save_dir /path/to/exported-hf/base \
  --model_assets_dir /path/to/original/daVinci-MagiHuman/base
```

导出的 7-shard safetensors、index、key、shape 和 dtype 与上游格式兼容。也可以预先将 HF
base 转成内部 DCP（例如统一使用 DCP 加载链路）：

```bash
mm-convert MagiHumanConverter hf_to_dcp \
  --hf_dir /path/to/daVinci-MagiHuman/base \
  --dcp_dir /path/to/daVinci-MagiHuman-base-dcp
```

不要用上游交错布局的权重直接加载内部布局的 DCP，反之亦然；跨布局交换一律通过
`MagiHumanConverter`。

## 环境变量声明

| 环境变量 | 描述 | 取值说明 |
| --- | --- | --- |
| `NON_MEGATRON` | 选择插件式 FSDP2 初始化路径 | 必须设为 `true` |
| `HCCL_CONNECT_TIMEOUT` | 分布式建链超时时间，单位为秒 | 样例为 `1200` |
| `PYTORCH_NPU_ALLOC_CONF` | NPU 缓存分配器配置 | 样例为 `expandable_segments:True` |
| `MULTI_STREAM_MEMORY_REUSE` | 多流内存复用 | 样例为 `2` |
| `TASK_QUEUE_ENABLE` | task queue 优化等级 | 样例为 `2` |
| `CPU_AFFINITY_CONF` | CPU 任务绑核配置 | 样例为 `1` |
