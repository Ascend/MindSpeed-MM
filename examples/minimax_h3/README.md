# MiniMax-H3 使用指南（FSDP2）

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [支持范围](#支持范围)
- [环境安装](#环境安装)
- [权重下载](#权重下载)
- [数据集准备及处理](#数据集准备及处理)
  - [任务与缓存格式](#任务与缓存格式)
  - [生成预计算缓存](#生成预计算缓存)
- [微调](#微调)
  - [配置文件](#配置文件)
  - [启动训练](#启动训练)
  - [正式训练配置](#正式训练配置)
- [开发与测试](#开发与测试)
  - [跨后端 BF16 反向精度注意事项](#跨后端-bf16-反向精度注意事项)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明

### 参考实现

MiniMax-H3 的 DiT 结构、权重命名、数据预处理和 Flow Matching 训练语义参考
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)：

```text
url=https://github.com/modelscope/DiffSynth-Studio.git
commit_id=0ad850953156c028c6334ea8c17c50032a1ef169
```

模型权重来自 [MiniMax/MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3)。

### 支持范围

当前实现使用 MindSpeed-MM FSDP2 后端进行 DiT 全参数微调，不包含 text encoder、
Video VAE 和 Audio VAE 的在线训练。训练数据需要先通过 DiffSynth-Studio 编码为
`.pth` 缓存。

| 任务 | 条件输入 | Transformer | 配置文件 | 启动脚本 |
| --- | --- | --- | --- | --- |
| T2VA | prompt | `FL2VA/transformer` | `minimax_h3_t2va.yaml` | `finetune_minimax_h3_t2va.sh` |
| FL2VA | prompt、首帧、尾帧 | `FL2VA/transformer` | `minimax_h3_fl2va.yaml` | `finetune_minimax_h3_fl2va.sh` |
| Ref2VA | prompt、图像/视频/音频 reference | `Ref2VA/transformer` | `minimax_h3_ref2va.yaml` | `finetune_minimax_h3_ref2va.sh` |

## 环境安装

请先按照 [MindSpeed-MM 安装指南](../../docs/zh/pytorch/install_guide.md) 安装匹配版本的 Ascend Driver、CANN、PyTorch 和 torch-npu。

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd MindSpeed-MM
bash scripts/install.sh --msbranch master
```

安装脚本检测到环境中已有 torch / torch_npu 且版本与目标版本不一致时，会交互式询问
是否重装（`Reinstall PyTorch? (y/n)`、`Reinstall torch_npu ...? (y/n)`）。如果当前
环境的 PyTorch 和 torch-npu 已按安装指南与驱动、CANN 匹配，应选择 `n` 跳过重装，
避免覆盖后与 CANN 不匹配；也可以在命令中追加 `-n` 参数自动跳过所有重装询问。

生成 MiniMax-H3 预计算缓存时还需要 DiffSynth-Studio，请根据DiffSynth-Studio 安装指南
安装依赖。(具体可以参考"DiffSynth-Studio/examples/minimax_h3/model_training"下的脚本)

## 权重下载

T2VA 与 FL2VA 共用 FL2VA 权重，Ref2VA 使用 Ref2VA 权重，因此三份示例 YAML 只需要
FL2VA 与 Ref2VA 两套权重。以下命令将两套权重一次性下载到示例 YAML 默认引用的
`./checkpoints/MiniMax-H3` 目录：

```bash
python -m pip install modelscope
modelscope download \
  --model MiniMax/MiniMax-H3 \
  --local_dir ./checkpoints/MiniMax-H3 \
  --include 'FL2VA/**' 'Ref2VA/**' LICENSE README.md
```

训练需要的核心目录如下：

```text
checkpoints/MiniMax-H3/
├── FL2VA/
│   ├── transformer/
│   ├── text_encoder/
│   ├── video_vae/
│   ├── audio_vae/
│   └── processor/
└── Ref2VA/
    ├── transformer/
    ├── text_encoder/
    ├── video_vae/
    ├── audio_vae/
    └── processor/
```

MindSpeed-MM 训练阶段只读取对应的 `transformer` 目录；其他组件由 DiffSynth-Studio
在数据预处理阶段使用。

## 数据集准备及处理

### 任务与缓存格式

三种任务的监督目标均为目标视频及其同步音频，条件输入不同：

| 任务 | Metadata | 预计算缓存中的任务特征 |
| --- | --- | --- |
| T2VA | CSV | 无 keyframe 或 reference anchor |
| FL2VA | CSV | 包含首尾帧条件 `keyframe_cond_anchor` |
| Ref2VA | JSON | 包含 `ref_blocks` 中的 visual/audio reference rows |

T2VA 和 FL2VA 的 CSV 至少需要以下字段：

```csv
video,prompt,input_audio
videos/example.mp4,"A person is speaking.",videos/example.mp4
```

Ref2VA 使用 JSON metadata，`references` 支持 `image`、`video`、`audio` 和
`video_audio`：

```json
[
  {
    "video": "videos/example.mp4",
    "input_audio": "videos/example.mp4",
    "prompt": "A person is speaking.",
    "references": [
      {"type": "image", "image": "references/example.png"},
      {
        "type": "video_audio",
        "video": "references/example.mp4",
        "audio": "references/example.mp4"
      }
    ]
  }
]
```

MiniMax-H3 使用 24 FPS；视频高度和宽度需要是 32 的倍数，帧数需要满足 `17n+5`。
修改任务、分辨率或帧数后必须重新生成缓存。

### 生成预计算缓存

使用 DiffSynth-Studio 的 `sft:data_process` 生成缓存。下面三个入口脚本位于
DiffSynth-Studio 仓库，分别用于生成对应任务的缓存：

```text
examples/minimax_h3/model_training/full/MiniMax-H3-T2VA.sh
examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh
examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA.sh
```

也可以直接调用 DiffSynth-Studio 的统一训练入口，并设置：

- `--task sft:data_process`；
- `--model_id_with_origin_paths` 指向对应任务的 text encoder、Video VAE 和 Audio VAE；
- T2VA 使用 `--extra_inputs input_audio`；
- FL2VA 使用 `--extra_inputs input_audio,input_image,end_image`；
- Ref2VA 使用 `--extra_inputs input_audio,references`，并使用 Ref2VA processor。

缓存目录结构可以包含 rank 子目录，MindSpeed-MM 会递归、按路径排序读取所有 `.pth`
文件：

```text
data/minimax_h3/t2va_cache/
├── 0/
│   ├── 0.pth
│   └── 1.pth
└── 1/
    ├── 0.pth
    └── 1.pth
```

## 微调

### 配置文件

三份示例配置用于单机 8 卡、50 个 optimizer step 的功能和数值验证。它们使用 BF16
前向、FP32 梯度归约、AdamW、固定数据顺序和固定随机输入。

| 配置项 | T2VA | FL2VA | Ref2VA |
| --- | ---: | ---: | ---: |
| `dit_num_layers` | 1 | 5 | 5 |
| `ulysses_parallel_size` | 1 | 2 | 2 |
| `gradient_accumulation_steps` | 1 | 2 | 2 |
| `dataset_repeat` | 4 | 4 | 40 |
| 8 卡 global batch | 8 | 8 | 8 |

运行前确认每份 YAML 中的以下路径与本地资产一致：

| 配置项 | 说明 |
| --- | --- |
| `data.dataset_param.basic_parameters.dataset_dir` | DiffSynth-Studio 生成的 `.pth` 缓存目录 |
| `model.model_name_or_path` | 包含 `config.json` 的 transformer 目录 |
| `training.load` | 初始 HF transformer 权重目录，应与 `model_name_or_path` 一致 |

### 启动训练

从 MindSpeed-MM 仓库根目录选择一个任务运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPUS_PER_NODE=8

bash examples/minimax_h3/finetune_minimax_h3_t2va.sh
# bash examples/minimax_h3/finetune_minimax_h3_fl2va.sh
# bash examples/minimax_h3/finetune_minimax_h3_ref2va.sh
```

脚本启动前会检查配置文件，训练日志默认写入 `logs/train_*.log`。可通过
环境变量覆盖配置和日志目录：

```bash
CONFIG_PATH=examples/minimax_h3/minimax_h3_ref2va.yaml \
LOG_DIR=logs/minimax_h3_ref2va \
bash examples/minimax_h3/finetune_minimax_h3_ref2va.sh
```

多机训练可设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。并行配置变化后，
需要重新计算 data parallel size 和梯度累积步数，以保持预期 global batch。

### 正式训练配置

示例 YAML 保留了减层和确定性设置，适合快速验证，不建议直接作为正式长程训练配置。
正式训练时复制对应 YAML，并按训练目标调整：

```yaml
data:
  dataset_param:
    basic_parameters:
      dataset_repeat: 1
  dataloader_param:
    shuffle: true

model:
  dit_num_layers: 50
  # 删除 loss_replay_seed

training:
  train_iters: 1000
  clip_grad: 1.0
  save: ./outputs/minimax_h3/checkpoints
  save_format: dcp
  save_interval: 100
  no_save_optim: false
  no_save_rng: false
```

`dit_num_layers` 应与训练目标一致。完整 MiniMax-H3 transformer 包含 50 个 DiT block；
减层仅用于资源受限的开发验证。

## 环境变量声明

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `ASCEND_ENV` | CANN 环境初始化脚本 | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `ASCEND_RT_VISIBLE_DEVICES` | 当前进程可见的 NPU 列表 | `0,1,2,3,4,5,6,7` |
| `NPUS_PER_NODE` | 每节点训练进程数 | `8` |
| `NNODES` | 节点数 | `1` |
| `NODE_RANK` | 当前节点编号 | `0` |
| `MASTER_ADDR` | 主节点地址 | `localhost` |
| `MASTER_PORT` | torchrun rendezvous 端口 | `6000` |
| `CONFIG_PATH` | 训练 YAML 路径 | 对应任务的默认 YAML |
| `LOG_DIR` | 训练日志目录 | `logs` |
| `HCCL_DETERMINISTIC` | HCCL 确定性计算开关 | `True` |
| `PYTORCH_NPU_ALLOC_CONF` | torch-npu 内存分配配置 | `expandable_segments:True` |

## 注意事项

- T2VA 和 FL2VA 必须加载 FL2VA transformer，Ref2VA 必须加载 Ref2VA transformer；
- 三种任务的缓存不可混用，不同 shape 的缓存也应使用不同目录；
- `model.model_name_or_path` 与 `training.load` 应指向同一份初始 transformer 权重；
- `ulysses_parallel_size > 1` 时需要使用 `flash_attention_2`；
- Ref2VA 缓存含有 PIL 对象，训练加载使用 `torch.load(weights_only=False)`；只能加载由可信
  DiffSynth-Studio 流程在本地生成的 `.pth` 缓存；
- 不要移除或将 AdaLN 调制函数中的 `.float()` 移到 `index_select` 之后；这会改变 BF16
  反向重复索引的累加精度，导致 GPU/NPU 训练轨迹分叉；
- 示例配置的 `loss_replay_seed`、`shuffle=false` 和 `clip_grad=0` 用于确定性验证，正式
  训练应根据实际任务重新配置；

### GPU\NPU BF16 反向精度注意事项

MiniMax-H3 的 AdaLN 调制会使用 `index_select` 从 `shift`、`scale` 和 `gate` 中选择
token 对应的参数行。前向阶段是 gather；反向阶段则需要将重复索引对应的梯度执行
index-add 归约。对 BF16 输入，NPU 的 `index_select` 反向会在归约过程中升精度到
FP32，而 GPU 通常按 BF16 累加，因此会出现以下差异：

- `NPU BF16` 的反向结果接近 `NPU FP32` 和 `GPU FP32`；
- `GPU BF16` 的反向结果可能不同；
- 差异主要出现在训练反向和 optimizer 更新，前向推理不受该反向归约差异影响。

为保持 GPU/NPU 训练精度对齐，AdaLN 调制路径需要在 `index_select` **之前**将参数转为
FP32，并在输出端恢复为输入 dtype。当前实现应保持如下形式：

```python
def _modulate_scale_shift(x, shift, scale, indices):
    return (
        x * (1.0 + scale.float().index_select(0, indices))
        + shift.float().index_select(0, indices)
    ).to(x.dtype)


def _modulate_gate(x, gate, other, indices):
    return (x + gate.float().index_select(0, indices) * other).to(x.dtype)
```

这里的 `.float()` 须位于 `index_select` 前面，才能使反向的重复索引归约使用 FP32
累加；不要求将整个 DiT block、Attention 或推理流程改为 FP32。
