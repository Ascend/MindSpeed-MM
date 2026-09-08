# LTX2.3 使用指南（FSDP2）

LTX-2.3 是 Lightricks 的 22B 音视频生成模型。本目录提供基于 MindSpeed-MM FSDP2 后端的全量微调示例，当前支持文本生成视频（t2v）和文本生成音视频（t2av）。

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [任务支持列表](#任务支持列表)
- [前置准备](#前置准备)
  - [下载模型文件](#1-下载模型文件)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [数据集准备及处理](#数据集准备及处理)
  - [数据格式](#1-数据格式)
  - [预处理数据](#2-预处理数据)
  - [预处理目录结构](#3-预处理目录结构)
- [训练](#训练)
  - [准备工作](#1-准备工作)
  - [启动训练](#2-启动训练)
- [环境变量声明](#环境变量声明)
- [常见问题](#常见问题)

## 版本说明

### 参考实现

【LTX-2.3 源仓】

```text
url=https://github.com/Lightricks/LTX-2.git
commit_id=d6053703e00195bc668cbd1d5eda9dc0b2e7b74a
```

【MindSpeed-MM 迁移】

```text
model_adapter_commit=c416a023085020df2b37dbef2cf75b72b66fb789
training_example_commit=f5efe355c06efcae1ed048a5d2fe3c8d5d51464f
```

MindSpeed-MM 开源仓不包含 LTX-2.3 的上游 `ltx_core` 和 `ltx_trainer` 代码。训练前需从上述指定版本的 LTX-2 源仓复制这两个目录到 MM 仓库；训练时会从复制后的目录加载代码。

### 变更记录

2026.07.16：首次支持 LTX-2.3（22B）模型的 t2v、t2av 全量微调训练，使用 FSDP2 后端。

## 任务支持列表

| 模型 | 模型规模 | t2v 全量微调 | t2av 全量微调 |
|------|:--------:|:------------:|:-------------:|
| LTX-2.3 | 22B | ✔ | ✔ |

> 当前示例仅覆盖训练，不包含推理和 LoRA 训练流程。

## 前置准备

训练前需要准备 LTX-2.3 基础检查点。数据预处理还需要 Gemma 文本编码器。

> [!NOTE]
>
> 如无法顺利访问 HuggingFace，推荐前往 ModelScope 获取对应文件，并确认下载文件来源和完整性。

### 1. 下载模型文件

| 模型 | 文件或目录 | 用途 | 下载地址 |
|------|------|------|------|
| **LTX-2.3 基础检查点** | `ltx-2.3-22b-dev.safetensors` | MM 训练和数据预处理 | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |
| **Gemma 3 文本编码器** | 完整模型目录 | 数据预处理时生成文本 embedding | [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) |

MM 训练阶段从 LTX-2.3 检查点中加载 embedding connector；Gemma 目录只在离线预处理 caption 时使用。空间/时间上采样器和 distilled LoRA 不属于本训练流程的必需文件。

## 环境安装

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考 [MindSpeed-MM 安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/install_guide.md)，完成昇腾驱动、CANN、PyTorch、`torch_npu` 和 MindSpeed 的安装。建议使用 Python 3.10；`torch`、`torch_npu`、CANN 版本需相互匹配。

```bash
conda activate <your_conda_env>
source <CANN_INSTALL_PATH>/set_env.sh
```

将 `<your_conda_env>` 和 `<CANN_INSTALL_PATH>` 替换为实际环境名称和 CANN 安装路径。预处理依赖 LTX-2 源仓，需安装 `uv`。训练环境还需按上游 `packages/ltx-core/pyproject.toml` 和 `packages/ltx-trainer/pyproject.toml` 补齐依赖，并保证其与 NPU 软件栈兼容；不要直接用上游 CUDA 环境替换已安装的 `torch_npu`。

### 2. 环境搭建

拉取两个代码仓并安装 MindSpeed-MM。以下命令假设两个仓库位于同一父目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/Lightricks/LTX-2.git
```

从 LTX-2 源仓获取与本示例匹配的代码；其他目录布局请相应调整复制命令。

```bash
cd LTX-2
git checkout d6053703e00195bc668cbd1d5eda9dc0b2e7b74a

# 在 LTX-2 源仓根目录准备预处理环境
uv sync --frozen

# 复制训练所需的上游模块到 MindSpeed-MM
cd ../MindSpeed-MM
pip install -e .
mkdir -p mindspeed_mm/fsdp/models/ltx2_3/vendor
cp -r ../LTX-2/packages/ltx-core/src/ltx_core \
  mindspeed_mm/fsdp/models/ltx2_3/
cp -r ../LTX-2/packages/ltx-trainer/src/ltx_trainer \
  mindspeed_mm/fsdp/models/ltx2_3/vendor/
```

如果目标目录已存在旧副本，请先确认其内容与指定 LTX-2 提交一致，避免混用不同版本的上游代码。复制完成后，应存在 `mindspeed_mm/fsdp/models/ltx2_3/ltx_core` 和 `mindspeed_mm/fsdp/models/ltx2_3/vendor/ltx_trainer`。这两个目录不纳入 MM 开源仓库版本控制；重新克隆仓库或清理工作区后，需要重新执行复制步骤。

## 数据集准备及处理

训练阶段读取离线生成的视频 latent、音频 latent 和文本 embedding，不直接读取原始视频。数据预处理脚本仍使用 LTX-2 源仓的 `packages/ltx-trainer/scripts/process_dataset.py`。

### 1. 数据格式

数据集元数据可以是 JSON、JSONL 或 CSV。下面以 JSON 为例，`media_path` 可以是相对于 `dataset.json` 的路径，也可以是绝对路径：

```json
[
  {
    "caption": "A woman with long brown hair sits at a wooden desk, typing on a laptop.",
    "media_path": "videos/video1.mp4"
  },
  {
    "caption": "A chef in a white uniform stands in a professional kitchen, carefully plating a gourmet dish.",
    "media_path": "videos/video2.mp4"
  }
]
```

字段说明：

- `caption`：用于 Gemma 文本编码器生成条件 embedding 的文本描述。
- `media_path`：目标视频或图片路径。t2av 训练时，视频需要包含可读取的音频轨道。

### 2. 预处理数据

假设数据集目录为 `/path/to/dataset`，源仓目录为 `/path/to/LTX-2`。先准备数据目录：

```text
/path/to/dataset/
├── dataset.json
└── videos/
    ├── video1.mp4
    └── video2.mp4
```

#### 2.1 t2v 预处理

```bash
cd /path/to/LTX-2/packages/ltx-trainer

uv run python scripts/process_dataset.py /path/to/dataset/dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --text-encoder-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

#### 2.2 t2av 预处理

在 t2v 命令的基础上增加 `--with-audio`：

```bash
cd /path/to/LTX-2/packages/ltx-trainer

uv run python scripts/process_dataset.py /path/to/dataset/dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --text-encoder-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
    --with-audio
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--resolution-buckets` | 分辨率桶，格式为 `宽度x高度x帧数`，多个桶使用 `;` 分隔 |
| `--model-path` | LTX-2.3 `.safetensors` 检查点 |
| `--text-encoder-path` | Gemma 文本编码器目录 |
| `--with-audio` | 生成 `audio_latents`，供 t2av 训练使用 |
| `--output-dir` | 指定预处理输出目录；默认写入数据集目录下的 `.precomputed` |
| `--decode` | 预处理完成后解码 latent，便于检查视频和音频 |
| `--overwrite` | 使用新的模型、分辨率或文本编码器时强制重新生成结果 |

分辨率桶需要满足 LTX VAE 的约束：宽度和高度是 32 的整数倍，帧数满足 `frames % 8 == 1`。例如 `960x544x49` 合法。使用多个分辨率桶时，必须保持训练 batch size 为 1；当前示例的 `micro_batch_size` 已设置为 1。

### 3. 预处理目录结构

`process_dataset.py` 默认生成如下目录：

```text
/path/to/dataset/
├── dataset.json
├── videos/
└── .precomputed/
    ├── latents/              # 视频 latent，t2v 和 t2av 都需要
    │   └── videos/*.pt
    ├── conditions/           # 文本条件 embedding 和 attention mask
    │   └── videos/*.pt
    └── audio_latents/        # 仅 t2av 需要
        └── videos/*.pt
```

每个目录中的 `.pt` 文件必须按相同的相对路径一一对应。`conditions/*.pt` 至少包含 `video_prompt_embeds` 和 `prompt_attention_mask`；音频训练还需要可用的 `audio_prompt_embeds`。如果 `data.dataset_param.basic_parameters.dataset_dir` 指向数据集根目录，MM 会自动发现其中的 `.precomputed`；也可以直接指向 `.precomputed` 目录。

## 训练

### 1. 准备工作

完成环境、模型和数据准备后，选择对应 YAML，在仓库根目录按实际情况修改以下参数：

- `data.dataset_param.basic_parameters.dataset_dir`：预处理数据根目录或 `.precomputed` 目录。
- `model.model_name_or_path`：LTX-2.3 模型目录或检查点路径。
- `model.checkpoint_path`：`ltx-2.3-22b-dev.safetensors` 路径。
- `training.load`：初始加载的基础检查点路径。
- `training.save`：训练输出目录；示例默认为空字符串，正式训练建议改为实际可写路径。
- `parallel`：根据实际 NPU 数量和集群拓扑调整并行参数。

两个配置文件分别对应：

| 配置文件 | 训练任务 | `with_audio` | 默认 NPU 数 |
|------|------|:------:|:------:|
| `examples/ltx2_3/ltx2_3_t2v.yaml` | 文生视频 | `false` | 8 |
| `examples/ltx2_3/ltx2_3_t2av.yaml` | 文生音视频 | `true` | 16 |

### 2. 启动训练

先进入 MindSpeed-MM 仓库根目录并加载环境：

```bash
cd /path/to/MindSpeed-MM
conda activate <your_conda_env>
source <CANN_INSTALL_PATH>/set_env.sh
```

#### t2v

默认单机 8 卡：

```bash
bash examples/ltx2_3/finetune_ltx2_3_t2v.sh
```

#### t2av

默认单机 16 卡：

```bash
bash examples/ltx2_3/finetune_ltx2_3_t2av.sh
```

脚本中的 `NPUS_PER_NODE`、`NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT` 均可通过环境变量覆盖。例如单机 8 卡启动 t2v：

```bash
NPUS_PER_NODE=8 NNODES=1 NODE_RANK=0 \
  bash examples/ltx2_3/finetune_ltx2_3_t2v.sh
```

训练日志保存在当前目录的 `logs/` 下，脚本会在训练结束后统计平均迭代时间和 samples/s。多机训练时请保证各节点能访问相同的模型和 `.precomputed` 路径，并为每个节点设置正确的 `NODE_RANK`。

## 环境变量声明

启动脚本默认设置以下环境变量：

| 环境变量 | 描述 | 当前示例值 |
|------|------|------|
| `NON_MEGATRON` | 使用非 Megatron 训练路径 | `true` |
| `TASK_QUEUE_ENABLE` | task queue 算子下发队列优化等级 | `2` |
| `CPU_AFFINITY_CONF` | CPU 端算子任务绑核 | `1` |
| `HCCL_CONNECT_TIMEOUT` | 分布式 socket 建链超时，单位为秒 | `1200` |
| `PYTORCH_NPU_ALLOC_CONF` | NPU 缓存分配器配置 | `expandable_segments:True` |
| `MULTI_STREAM_MEMORY_REUSE` | 多流内存复用开关或等级 | `2` |
| `NPUS_PER_NODE` | 每节点参与训练的 NPU 数 | t2v 为 `8`，t2av 为 `16` |
| `NNODES` | 节点数 | `1` |
| `NODE_RANK` | 当前节点编号，从 `0` 开始 | `0` |

## 常见问题

1. **提示找不到 `ltx_core` 或 `ltx_trainer`**：确认已从指定 LTX-2 提交复制 `ltx_core` 和 `ltx_trainer`，且当前 MM 仓库存在 `mindspeed_mm/fsdp/models/ltx2_3/ltx_core` 和 `mindspeed_mm/fsdp/models/ltx2_3/vendor/ltx_trainer`，并从 MM 仓库根目录启动训练。
2. **提示缺少 `audio_latents` 或样本数量不一致**：t2av 配置要求每个 `latents/*.pt` 都有对应的 `audio_latents/*.pt`，请重新执行带 `--with-audio` 的预处理命令。
3. **出现 batch shape mismatch**：同一 batch 中的样本必须具有相同的 latent 形状。使用多个分辨率桶时将 `micro_batch_size` 保持为 1，并确保帧数和分辨率桶合法。
4. **训练能够启动但没有输出权重**：检查 YAML 中的 `training.save`，当前示例为 `""`，需要改为实际输出目录后才会保存训练检查点。
5. **多机启动失败**：检查所有节点的 CANN 环境、`MASTER_ADDR`/`MASTER_PORT`、`NODE_RANK` 和模型/数据路径，并确认端口可达。
