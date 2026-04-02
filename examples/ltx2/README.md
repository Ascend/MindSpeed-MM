# LTX2 使用指南（FSDP2）

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [前置准备](#前置准备)
  - [下载模型文件](#1-下载模型文件)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [数据集准备及处理](#数据集准备及处理)
  - [数据格式（ltx2_precomputed）](#1-数据格式ltx2_precomputed)
    - [数据集元数据格式](#11-数据集数据格式)
  - [预处理脚本（生成 .precomputed）](#2-预处理脚本生成-precomputed)
    - [基本预处理（仅视频，t2v）](#21-基本预处理仅视频t2v)
    - [带音频的预处理（t2av）](#22-带音频的预处理t2av)
    - [预处理目录结构](#23-预处理目录结构)
- [训练](#训练)
  - [准备工作](#1-准备工作)
  - [配置文件说明](#2-配置文件说明)
  - [启动训练（t2v / t2av）](#3-启动训练t2v--t2av)
- [环境变量声明](#环境变量声明)

## 版本说明

### 参考实现

upstream=LTX2/packages/ltx-trainer

### 变更记录

2026.03.24: 首次支持 LTX2.0 模型的 t2v、t2av 微调训练，支持 fsdp2 后端

---
<a id="jump0"></a>

## 前置准备

训练前需要下载以下模型文件：

<a id="jump0.1"></a>

### 1. 下载模型文件

| 模型 | 文件 | 下载地址 |
|------|------|----------|
| **LTX-2 检查点** | `ltx-2-19b-dev.safetensors`  | [HuggingFace Hub - Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) |
| **Gemma 文本编码器** | 完整模型目录 | [HuggingFace Hub - google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/) |

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)，完成昇腾软件安装。
> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

‼️部分特性依赖较新版本的torch_npu和CANN，推荐使用以下版本

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler)
- [torch_npu](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)

<a id="jump1.2"></a>

### 2. 环境搭建

```bash
# 1. 克隆 MindSpeed-MM
git clone https://gitcode.com/Ascend/MindSpeed-MM.git

# 2. 安装 mindspeed 及依赖（如已集成可跳过）
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/
cd ../MindSpeed-MM

# 3. 克隆 LTX-2 源仓（用于获取 ltx_core 模块）
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
git checkout 28c3c73fe557666c3de176e1e50a5220152ccfca

# 4. 复制 ltx_core 模块到 MindSpeed-MM
cp -r packages/ltx-core/src/ltx_core ../MindSpeed-MM/mindspeed_mm/fsdp/models/ltx2/

# 5. 安装 mindspeed mm
cd ..
pip install -e .
```

---
<a id="jump2"></a>

## 数据集准备及处理

离线将视频/音频编码为 latent，并将文本编码为 prompt embedding，训练阶段直接读取 `.precomputed/*` 加速训练与减少线上依赖。

<a id="jump2.1"></a>

### 1. 数据格式（ltx2_precomputed）

### 1.1 数据集数据格式 <a id="jump2.1.2"></a>

**`dataset.json` JSON 格式示例：**

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

**注意：**

- `caption`：视频描述文本，用于文本编码器生成条件嵌入
- `media_path`：视频文件路径，支持相对路径或绝对路径

<a id="jump2.2"></a>

### 2. 预处理脚本（生成 .precomputed）

预处理脚本位于源仓 `LTX-2/packages/ltx-trainer/scripts/` 目录下。本框架依赖源仓脚本进行预处理。

### 2.1 基本预处理（仅视频，t2v）

```bash
cd /path/to/LTX-2/packages/ltx-trainer
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model
```

### 2.2 带音频的预处理（t2av）

```bash
cd /path/to/LTX-2/packages/ltx-trainer
uv run python scripts/process_dataset.py dataset.json \
    --resolution-buckets "960x544x49" \
    --model-path /path/to/ltx-2-model.safetensors \
    --text-encoder-path /path/to/gemma-model \
    --with-audio
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `dataset.json` | 数据集元数据文件路径（位置参数） |
| `--model-path` | LTX-2 模型检查点（.safetensors） |
| `--text-encoder-path` | Gemma 文本编码器目录 |
| `--resolution-buckets` | 分辨率桶配置，格式 `宽度x高度x帧数` |
| `--with-audio` | 启用音频处理（t2av 训练） |
| `--lora-trigger` | LoRA 触发词（推理时需包含） |
| `--decode` | 解码验证（生成视频预览） |

### 2.3 预处理目录结构 <a id="jump2.1.1"></a>

`ltx2_precomputed` 默认读取 `preprocessed_data_root` 下的 `.precomputed` 子目录：

```text
${preprocessed_data_root}/
├── dataset.json              # 数据集元数据文件
├── videos/                   # 原始视频文件（可选）
└── .precomputed/             # 预处理结果（自动生成）
    ├── latents/              # 视频潜在表示
    │   └── *.pt
    ├── conditions/           # 文本嵌入
    │   └── *.pt
    └── audio_latents/        # 音频潜在表示（t2av 训练时）
        └── *.pt
```

---
<a id="jump3"></a>

## 训练

<a id="jump3.1"></a>

### 1. 准备工作

在配置文件中需要正确设置以下路径：

- `model.model_name_or_path` / `model.checkpoint_path`：LTX2 safetensors 权重路径
- `model.text_encoder_path`：Gemma text encoder 目录（如需使用 connector）
- `data.dataset_param.basic_parameters.dataset_dir`：预处理数据根目录（包含 `.precomputed/`）

并确保 `training.plugin` 包含：

- `mindspeed_mm/fsdp/models/ltx2/ltx2_fsdp2`
- `mindspeed_mm/fsdp/data/datasets/ltx2`

<a id="jump3.2"></a>

### 2. 配置文件说明

关键配置项：

```yaml
model:
  model_name_or_path: "/path/to/ltx-2-model.safetensors"
  checkpoint_path: "/path/to/ltx-2-model.safetensors"
  text_encoder_path: "/path/to/gemma-model"

data:
  dataset_param:
    basic_parameters:
      dataset_dir: "/path/to/preprocessed/data"

training:
  plugin:
    - mindspeed_mm/fsdp/models/ltx2/ltx2_fsdp2
    - mindspeed_mm/fsdp/data/datasets/ltx2

output_dir: "outputs/my_training_run"
```

<a id="jump3.3"></a>

### 3. 启动训练（t2v / t2av）

t2v（text-to-video）：

```bash
bash examples/ltx2/finetune_ltx2_t2v.sh
```

t2av（text-to-audio-video）：

```bash
bash examples/ltx2/finetune_ltx2_t2av.sh
```

---
<a id="jump4"></a>

## 环境变量声明

| 环境变量                    | 描述                                         | 取值说明 |
|-----------------------------|----------------------------------------------|----------|
| `TASK_QUEUE_ENABLE`         | 控制 task_queue 算子下发队列优化等级          | `0` 关闭；`1` 开启 Level 1；`2` 开启 Level 2 |
| `CPU_AFFINITY_CONF`         | 控制 CPU 端算子任务绑核                       | `0` 关闭；`1` 粗粒度；`2` 细粒度 |
| `HCCL_CONNECT_TIMEOUT`      | 分布式 socket 建链超时等待时间（单位 s）      | `[120, 7200]`，默认 `120` |
| `PYTORCH_NPU_ALLOC_CONF`    | NPU 缓存分配器行为                            | 例如 `expandable_segments:True` |
| `MULTI_STREAM_MEMORY_REUSE` | 多流内存复用开关                              | `0` 关闭；`1` 开启 |
