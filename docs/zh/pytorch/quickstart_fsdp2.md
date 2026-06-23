# 快速上手（FSDP2 后端）

本文档面向 **FSDP2 后端**，以多模态理解模型 **Qwen3-VL** 在 COCO2017 上的微调为例，引导完成"下载 → 转换 → 配置 → 启动"的完整流程。

> 本文以 Qwen3-VL 的 FSDP2 样例为基准，其它模型的训练流程与此类似。

## FSDP2 后端简介

FSDP2（Fully Sharded Data Parallel 2）沿数据并行维对参数、梯度、优化器状态做全分片，显著降低单卡显存占用，且与模型结构解耦、适配新模型成本低。在 MindSpeed-MM 中，FSDP2 训练具有以下特点：

- **训练器**：`mindspeed_mm/fsdp/train/trainer.py`，通过 `torchrun` 即可启动；
- **配置集中**：一份 YAML 即可，分 `parallel`/`data`/`model`/`features`/`training`/`tools` 六段；
- **权重格式**：使用 DCP（PyTorch 分布式检查点）格式，配合 meta init 初始化降低峰值显存；
- **可扩展**：在分片基础上可叠加 Ulysses 上下文并行、MoE 专家并行等。

## 总体流程

```text
Step1 环境安装 → Step2 权重下载+转 DCP → Step3 数据准备 → Step4 配置 YAML → Step5 启动微调
```

## Step 1：环境安装

先参考 [安装指导](./install_guide.md) 完成昇腾基础软件安装。推荐配套版本：

- Python 3.10
- CANN 8.5.2
- torch / torch_npu 2.7.1

然后拉取代码并安装：

```bash
# 拉取 MindSpeed-MM 代码仓
git clone https://gitcode.com/Ascend/MindSpeed-MM.git

# 安装 mindspeed 及依赖
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/

# 安装 mindspeed mm 及依赖
cd ../MindSpeed-MM
pip install -e .
```

## Step 2：权重下载及转换（转 DCP 格式）

1. **下载权重**：从 HuggingFace 下载 [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)，保存到 `ckpt/Qwen3-VL-30B-A3B-Instruct`。

2. **转换为 DCP 格式**：使用 meta init 初始化时需要 DCP 权重（**Qwen3-VL-30B / 235B 必须使用 meta init，仓库默认开启**）：

   ```bash
   mm-convert GenericDCPConverter hf_to_dcp \
     --hf_dir ckpt/Qwen3-VL-30B-A3B-Instruct \
     --dcp_dir ckpt/Qwen3-VL-30B-A3B-Instruct-dcp
   ```

   转换后 `--dcp_dir` 下会生成 `release/` 文件夹和 `latest_checkpointed_iteration.txt`。随后在配置文件中开启 `init_model_with_meta_device`，并将 **`load` 指向该 dcp 目录**（即 `release` 的上一级目录 `ckpt/Qwen3-VL-30B-A3B-Instruct-dcp`），详见 Step 4。

## Step 3：数据集准备及处理

以 COCO2017 为例：

1. 下载 [COCO2017](https://cocodataset.org/#download)，解压到 `./data/COCO2017`；
2. 下载描述文件 [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) 到 `./data/`；
3. 运行数据转换脚本：

   ```bash
   # 该脚本对 Qwen2-VL / Qwen3-VL 通用，可直接运行
   python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py
   ```

   该脚本**无需传参，直接运行即可**；但脚本内的读写路径是写死的：

   - 读取标注：`./data/llava_instruct_150k.json`
   - 读取图片：`./data/COCO2017/train2017/`
   - 输出结果：`./data/mllm_format_llava_instruct_data.json`

   因此请在**仓库根目录**下运行，并确保数据已按上述路径放好；若数据放在别处，需先修改脚本内对应的路径。转换后的目录结构如下：

   ```text
   ├── data
       ├── COCO2017
           └── train2017
       ├── llava_instruct_150k.json
       └── mllm_format_llava_instruct_data.json
   ```

## Step 4：配置参数

只需编辑**一份**配置文件 `examples/qwen3vl/qwen3vl_30B_config_v1.yaml`，它由六个顶层段组成：`parallel`（并行/分片）、`data`（数据）、`model`（模型/冻结）、`features`（loss 等）、`training`（训练超参/权重收发）、`tools`（工具）。

### 4.1 必改字段

| 参数所在段 | 参数 | 改成 |
|---|---|---|
| `data` | `model_name_or_path` | HF 模型目录 `./ckpt/Qwen3-VL-30B-A3B-Instruct`，用于加载 tokenizer 和模型 config（**非训练权重**；训练权重由 `training.load` 指定） |
| `data` | `dataset_dir` | 数据集根目录，即 `./data` |
| `data` | `dataset` | Step 3 预处理后的 `./data/mllm_format_llava_instruct_data.json` |
| `training` | `load` | Step 2 转换出的 **DCP 权重**路径 `./ckpt/Qwen3-VL-30B-A3B-Instruct-dcp`（默认注释，meta init 时取消注释并填写） |
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

### 4.2 分片策略（`parallel` 段，样例已配好）

FSDP 的分片配置在该 YAML 的 `parallel` 段，常规微调无需改动：

```yaml
parallel:
  tensor_parallel_size: 1
  fully_shard_parallel_size: auto    # FSDP 全分片组大小，auto 按全局卡数自动设定
  fsdp_plan:
    apply_modules:                   # 需要 fully_shard 分片的模块
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}
      - lm_head
      # ……（完整列表见样例文件）
    param_dtype: bf16
    reduce_dtype: fp32
  ulysses_parallel_size: 1           # 长序列时可 >1，开启 Ulysses 上下文并行
  expert_parallel_size: 1            # MoE 专家并行
```

## Step 5：启动微调

1. 按机器规模配置启动脚本 `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh`：

   ```bash
   NPUS_PER_NODE=16        # 单机卡数
   MASTER_ADDR=localhost   # 多机时改为主节点 IP
   MASTER_PORT=6000
   NNODES=1                # 多机时改为节点总数
   NODE_RANK=0             # 多机时改为本机节点序号
   ```

2. 启动：

   ```bash
   bash examples/qwen3vl/finetune_qwen3vl_30B_v1.sh
   ```

   日志输出到 `logs/`，权重保存到 YAML 中 `training.save` 指定的目录。

> **loss 计算方式**：`features` 段的 `loss_type` 默认 `default` 即可；如需按样本（per-sample）或按 token（per-token）归一等自定义方式，再调整该字段。

## （可选）后续处理

训练得到的是 DCP 格式权重，若需用 HuggingFace/Transformers 加载，可先导回 HF 格式。

### 1. 权重导出（DCP → HF）

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

### 2. LoRA 微调

如需以更低显存开销进行微调，可改用 LoRA（参考 [LoRA 微调（FSDP2）](../features/lora_finetune_fsdp2.md)）。

## 延伸阅读（可选）

完成上述步骤已可跑通微调，以下内容按需查阅：

- 把自己的模型接入 FSDP2：[FSDP2 模型迁移开发指南](../features/fsdp2_developer_migration_guide.md)
- 配置字段说明：[Config 中心使用手册](../config.md)
- 原理：[FSDP2 特性说明](../features/fsdp2.md)
- 高效微调：[LoRA 微调（FSDP2）](../features/lora_finetune_fsdp2.md)、[Agentic SFT](../features/agentic_sft.md)
- loss 计算方式：[VLM Loss 计算类型](../features/vlm_model_loss_calculate_type.md)
- 遇到问题：[故障排查](../troubleshooting.md)
