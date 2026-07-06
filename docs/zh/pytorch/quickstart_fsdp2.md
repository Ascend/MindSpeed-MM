# 快速入门：Qwen3-VL模型微调

MindSpeed MM支持多模态理解模型，下面分别以Qwen3-VL模型为例，介绍MindSpeed MM的使用方法，引导开发者快速上手预置模型在昇腾NPU上的高效运行。

## 多模态理解模型

本章节以Qwen3-VL-30B为例，指导用户在单机场景下如何完成多模态理解模型的微调。

FSDP2（Fully Sharded Data Parallel 2）沿数据并行维对参数、梯度、优化器状态做全分片，显著降低单卡显存占用，且与模型结构解耦、适配新模型成本低。在 MindSpeed-MM中，FSDP2训练具有以下特点：

- **训练器**：`mindspeed_mm/fsdp/train/trainer.py`，启动脚本一行`torchrun`即可启动；
- **配置集中**：一份 YAML 即可，分`parallel`/`data`/`model`/`features`/`training`/`tools`六段；
- **权重格式**：使用 DCP（PyTorch 分布式检查点）格式，配合 meta init 初始化降低峰值显存；
- **可扩展**：在分片基础上可叠加 Ulysses 上下文并行、MoE 专家并行等。

## 环境准备

1. 基于PyTorch框架和Python3.10完成模型训练环境的安装，具体请参见[MindSpeed MM安装指导](install_guide.md)。
2. 在`MindSpeed-MM`下创建以下目录用于存储日志、数据及权重文件。

    ```bash
    mkdir logs
    mkdir data
    mkdir ckpt
    ```

## 权重下载及转换

1. 权重下载

   从Hugging Face下载对应的模型权重[Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)。
    >[!NOTE]
    >
    >如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

2. 权重文件保存

   创建`ckpt/Qwen3-VL-30B-A3B-Instruct`目录并将下载的模型权重保存到该目录下。

3. 权重转换

    可使用`mm-convert`工具对原始微调权重进行转换。执行如下命令运行工具：

    ```bash
    # Qwen3-VL-30B
    mm-convert GenericDCPConverter hf_to_dcp \
      --hf_dir ckpt/Qwen3-VL-30B-A3B-Instruct \
      --dcp_dir ckpt/Qwen3-VL-30B-A3B-Instruct-dcp
    ```

    **表 1** 权重转换工具参数解析

    |参数|说明|是否必选|默认值|
    |-|-|-|-|
    |GenericDCPConverter|Qwen3-VL模型转换工具|是|/|
    |hf_to_dcp|Hugging Face模型转换MindSpeed MM模型权重|是|/|
    |dcp_dir|转换后保存目录|是|/|
    |hf_dir|Hugging Face权重目录|是|/|

    > [!NOTE]
    > 由于Qwen3_VL和Qwen2_VL在权重转换逻辑上保持一致，更多工具详情可参见[权重转换命令行工具](../features/mm_convert.md)。

    使用 meta init 初始化时需要 DCP 权重（**Qwen3-VL-30B / 235B 必须使用 meta init，仓库默认开启**）
   转换后 `--dcp_dir` 下会生成 `release/` 文件夹和 `latest_checkpointed_iteration.txt`。随后在配置文件中开启 `init_model_with_meta_device` 并把 **`load` 指向该 dcp 目录**（写到 `release` 的上一级，即 `ckpt/Qwen3-VL-30B-A3B-Instruct-dcp`）。

## 数据集准备及处理

1. 数据集下载

   以COCO2017数据集为例，创建`data/COCO2017`目录后下载并解压[COCO2017](https://cocodataset.org/#download)数据集。
    >[!NOTE]
    >
    >如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

2. 获取数据集描述文件

   从Hugging Face下载图片数据集的描述文件[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)，保存至./data/路径下。

3. 数据集预处理

    执行如下数据转换脚本：

    ```python
    # 该脚本对 Qwen2-VL / Qwen3-VL 通用，可直接运行
    python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py
    ```

    转换后参考数据目录结构如下：

    ```bash
    ├── data
        ├── COCO2017
            ├── train2017
        ├── llava_instruct_150k.json
        ├── mllm_format_llava_instruct_data.json
        ...
    ```

## 配置参数

只需编辑**一份**配置文件 `examples/qwen3vl/qwen3vl_30B_config_v1.yaml`，它由六个顶层段组成：`parallel`（并行/分片）、`data`（数据）、`model`（模型/冻结）、`features`（loss 等）、`training`（训练超参/权重收发）、`tools`（工具）。

### 必改字段

| 所在段 | 字段 | 改成 |
|---|---|---|
| `data` | `model_name_or_path` | 转换前的**原始 HF 权重**路径，即 `./ckpt/Qwen3-VL-30B-A3B-Instruct` |
| `data` | `dataset_dir` | 数据集根目录，即 `./data` |
| `data` | `dataset` | 预处理后的 `./data/mllm_format_llava_instruct_data.json` |
| `training` | `load` | 转换出的 **DCP 权重**路径 `./ckpt/Qwen3-VL-30B-A3B-Instruct-dcp`（默认注释，meta init 时取消注释并填写） |
| `training` | `init_model_with_meta_device` | `true`（默认已开；30B/235B 必须） |
| `training` | `save` / `save_interval` | 权重保存路径与间隔 |
| `model` | `freeze` | 需冻结的模块（可选） |
| `features` | `loss_type` | loss 计算方式（`default` 等，见下） |

数据段示例（`model_name_or_path` 用转换前的原始 HF 路径，多机时 `cache_dir` 不要用同一挂载目录）：

```yaml
data:
  dataset_param:
    dataset_type: huggingface
    preprocess_parameters:
      model_name_or_path: &HF_MODEL_LOAD_PATH ./ckpt/Qwen3-VL-30B-A3B-Instruct
    basic_parameters:
      dataset_dir: ./data
      dataset: &DATASET_PATH ./data/mllm_format_llava_instruct_data.json
      cache_dir: ./data/cache_dir
```

### 分片策略（`parallel` 段，样例已配好）

FSDP 的分片配置在该 YAML 的 `parallel` 段，常规微调无需改动：

```yaml
parallel:
  tensor_parallel_size: 1
  fully_shard_parallel_size: auto    # FSDP全分片组大小，auto按全局卡数自动设定
  fsdp_plan:
    apply_modules:                   # 需要fully_shard分片的模块
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}
      - lm_head
      # ……（完整列表见样例文件）
    param_dtype: bf16
    reduce_dtype: fp32
  ulysses_parallel_size: 1           # 长序列时可大于1，开启Ulysses上下文并行
  expert_parallel_size: 1            # MoE专家并行
```

## 启动微调

1. 按机器规模配置启动脚本 `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh`：

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh

   NPUS_PER_NODE=16        # 单机卡数(MindSpeed MM支持Ascend 950 系列产品、Atlas A3 训练系列产品和Atlas A2 训练系列产品，且要求单NPU的片上内存为64GB及以上。当前示例脚本中NPUS_PER_NODE=16 表示需要16个NPU，如果实际情况低于此配置，可能遇到OOM问题)
   MASTER_ADDR=localhost   # 多机时改为主节点IP
   MASTER_PORT=6000
   NNODES=1                # 多机时改为节点总数
   NODE_RANK=0             # 多机时改为本机节点序号
   ```

2. 启动：

   ```bash
   bash examples/qwen3vl/finetune_qwen3vl_30B_v1.sh
   ```

   日志输出到 `logs/`，权重保存到 YAML 中 `training.save` 指定的目录。

>[!NOTE]
>
> **loss 计算方式**：`features` 段的 `loss_type` 默认 `default` 即可；如需按样本（per-sample）或按 token（per-token）归一等自定义方式，再调整该字段。

## 后续处理

训练得到的是 DCP 格式权重，若需用 HuggingFace/Transformers 加载，可先导回 HF 格式。

以下是dcp2hf的转换示例：

```bash
mm-convert GenericDCPConverter dcp_to_hf \
  --load_dir save_dir/release \
  --save_dir save_dir_hf \
  --model_assets_dir ./ckpt/Qwen3-VL-30B-A3B-Instruct
```

- `--load_dir`：训练保存目录下存放 DCP 分片的目录（`training.save` 路径下的 `release`，按实际保存结构填写）；
- `--save_dir`：导出的 HF 权重输出目录；
- `--model_assets_dir`：原始 HF 权重目录，用于复制 `config`/`tokenizer` 等资产。

> 完整参数以 `mm-convert GenericDCPConverter dcp_to_hf -h` 为准。

如只想低成本微调，可改用 LoRA（可参考 [LoRA 微调（FSDP2）](../features/lora_finetune_fsdp2.md)）。

## 支持 FSDP2 的模型样例

下列模型提供了可参照本文流程的 FSDP2 微调样例，具体命令可查阅各自 `examples/<模型>/` 下的 README：

| 模型 | 大致任务 | 样例目录 |
|---|---|---|
| Qwen3-VL | 图文理解（MoE） | `examples/qwen3vl/` |
| Qwen3.5 系列 | 图文理解 | `examples/qwen3_5/` |
| Qwen3.6 系列 | 图文理解 | `examples/qwen3_6/` |
| Qwen3-Omni | 全模态 | `examples/qwen3omni/` |
| Qwen3-TTS | 语音合成 | `examples/qwen3tts/` |
| Kimi-K2.5 | 多模态 | `examples/kimik2_5/` |
| LTX2 | 视频生成 | `examples/ltx2/` |
| MiMo-V2.5 | 多模态 | `examples/mimov2_5/` |
| CosyVoice3 | 语音 | `examples/cosyvoice3/` |
| FunASR | 语音识别 | `examples/funasr/` |
