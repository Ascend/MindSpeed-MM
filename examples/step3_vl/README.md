# Step3-VL 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [权重准备](#权重准备)
  - [模型下载](#1-模型下载)
  - [当前接入说明](#2-当前接入说明)
- [数据集准备及处理](#数据集准备及处理)
  - [数据格式](#1-数据格式)
  - [示例数据下载及处理](#2-示例数据下载及处理)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明

### 参考实现

```shell
url=https://huggingface.co/stepfun-ai/Step3-VL-10B
```

### 变更记录

2026.05.28: 首次支持 Step3-VL FSDP2 接入文档

---

## 环境安装

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。
> Python 版本推荐 3.10，torch 和 torch_npu 版本推荐 2.7.1。

### 2. 环境搭建

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git

# 安装mindspeed及依赖
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/

# 安装mindspeed mm及依赖
cd ../MindSpeed-MM
pip install -e .
```

---

## 权重准备

### 1. 模型下载

从 Hugging Face 下载 Step3-VL-10B 模型：

- [stepfun-ai/Step3-VL-10B](https://huggingface.co/stepfun-ai/Step3-VL-10B)

### 2. 当前接入说明

当前 Step3-VL 接入走本地 Hugging Face 目录的 `trust_remote_code` 加载方式。

需要保证以下文件保留在模型目录中：

```shell
config.json
configuration_step_vl.py
modeling_step_vl.py
processing_step3.py
tokenizer_config.json
tokenizer.json
vocab.json
merges.txt
model-*.safetensors
```

---

## 数据集准备及处理

### 1. 数据格式

当前示例复用 `dataset_type: huggingface` 数据链路，数据格式使用本地多模态 ShareGPT 风格字段：

- 图片字段：`images`
- 对话字段：`messages`
- 角色字段：`role`
- 内容字段：`content`

示例：

```json
{
  "images": ["train2017/000000000009.jpg"],
  "messages": [
    {"role": "user", "content": "<image>请描述图片内容。"},
    {"role": "assistant", "content": "这是一张..."}
  ]
}
```

### 2. 示例数据下载及处理

以下以 COCO2017 数据集为例。

(1) 用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2) 获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

(3) 运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py，转换后参考数据目录结构如下：

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

当前示例配置：

```shell
dataset_dir=/path/to/datasets
dataset=/path/to/datasets/jsonl
```

如果 JSON 中图片路径为相对路径，`dataset_dir` 需要指向图片根目录。

---

## 微调

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重准备**、**数据集准备及处理**。

### 2. 配置参数

根据实际情况修改 `examples/step3_vl/step3_vl_10b_config.yaml` 中的以下字段：

```yaml
data:
  dataset_param:
    preprocess_parameters:
      model_name_or_path: ./ckpt/hf_path/Step3-VL-10B
      trust_remote_code: true
    basic_parameters:
      dataset_dir: /path/to/datasets
      dataset: /path/to/datasets/jsonl
      template: step3_vl

  dataloader_param:
    collate_param:
      model_name: step3_vl

model:
  model_id: step3_vl
  model_name_or_path: ./ckpt/hf_path/Step3-VL-10B
  trust_remote_code: true

training:
  plugin:
    - mindspeed_mm/fsdp/models/step3_vl
    - mindspeed_mm/fsdp/data/datasets/huggingface
```

可按需要调整：

- `training.train_iters`
- `training.save_interval`
- `model.freeze`

### 3. 启动微调

修改 `examples/step3_vl/finetune_step3_vl_10b.sh` 中的环境变量后执行：

```bash
bash examples/step3_vl/finetune_step3_vl_10b.sh
```

---

## 环境变量声明

Step3-VL 示例脚本中使用了以下环境变量：

| 环境变量 | 描述 | 取值说明 |
| --- | --- | --- |
| `NON_MEGATRON` | 选择 MindSpeed-MM FSDP2 插件式训练入口 | `true`：启用非 Megatron 训练路径 |
| `HCCL_CONNECT_TIMEOUT` | 分布式场景下设备间 socket 建链超时时间 | 整数，取值范围 `[120,7200]`，单位 `s` |
| `PYTORCH_NPU_ALLOC_CONF` | 控制 NPU 缓存分配器行为 | `expandable_segments:True`：启用虚拟内存扩展段 |
| `MULTI_STREAM_MEMORY_REUSE` | 控制多流内存复用 | `0`：关闭；`1` 或 `2`：启用对应等级的复用策略 |
| `TASK_QUEUE_ENABLE` | 控制 task_queue 算子下发队列优化等级 | `0`：关闭；`1`：开启 Level 1；`2`：开启 Level 2 |
| `CPU_AFFINITY_CONF` | 控制 CPU 端算子任务处理器亲和性 | `0` 或未设置：关闭；`1`：粗粒度绑核；`2`：细粒度绑核 |

---

## 注意事项

1. `model.model_name_or_path` 和 `data.dataset_param.preprocess_parameters.model_name_or_path` 需要同时指向同一个 Step3-VL 本地目录。
2. 模型插件从启动 YAML 的 `model.model_name_or_path` 读取 remote code 路径。
3. 当前实现依赖本地 remote code，模型目录中的 `modeling_step_vl.py`、`processing_step3.py` 等文件不能删除。
4. 当前接入只覆盖图片 SFT 路径，不支持视频和音频输入。
5. 运行前请按实际环境修改 `examples/step3_vl/finetune_step3_vl_10b.sh` 中的 `set_env.sh`、`NPUS_PER_NODE` 等参数。
