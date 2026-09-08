# MiniMax-M3-VL 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [权重下载及转换](#权重下载及转换)
  - [权重下载](#1-权重下载)
  - [权重加载](#2-权重加载)
  - [权重保存](#3-权重保存)
- [数据集准备及处理](#数据集准备及处理)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明

### 参考实现

```shell
url=https://huggingface.co/MiniMaxAI/MiniMax-M3
transformers_version=v5.12.0
```

### 变更记录

2026.07.17: 首次支持MiniMax-M3-VL模型

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。

推荐使用 Python 3.10、PyTorch 2.7.1、torch_npu 2.7.1 和 Transformers 5.12.0。

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

执行如下指令安装：

```bash
bash scripts/install.sh --msbranch master && pip install "transformers==5.12.0"
```

如需使用 `minimax_m3_msa`，还需安装与当前 CANN 版本匹配且包含 MiniMax Sparse Attention 算子的 `cann_ops_transformer`。默认的 `flash_attention_2` 不依赖该组件。

---
<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

### 1. 权重下载

从HuggingFace下载[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)模型权重，并保存至本地目录，例如 `ckpt/hf_path/MiniMax-M3`。

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，可前往可信镜像下载，并注意核对文件的正确性与安全性。

<a id="jump2.2"></a>

### 2. 权重加载

MiniMax-M3-VL 的 NPU MoE 实现使用融合后的 gate/up 权重和堆叠的 expert 权重。加载 DCP 权重前，需使用专用转换器完成权重转换：

```bash
mm-convert MiniMaxM3Converter hf_to_dcp \
  --hf_dir ckpt/hf_path/MiniMax-M3 \
  --dcp_dir ckpt/dcp_path/MiniMax-M3-dcp
```

转换后的目录结构如下：

```text
MiniMax-M3-dcp
├── release
└── latest_checkpointed_iteration.txt
```

在 `minimax_m3_config.yaml` 中配置原始 Hugging Face 目录和转换后的 DCP 目录：

```yaml
data:
  dataset_param:
    preprocess_parameters:
      model_name_or_path: ./ckpt/hf_path/MiniMax-M3

model:
  model_name_or_path: ./ckpt/hf_path/MiniMax-M3

training:
  init_model_with_meta_device: true
  load: ./ckpt/dcp_path/MiniMax-M3-dcp
  load_format: dcp
```

<a id="jump2.3"></a>

### 3. 权重保存

通过 `training.save` 和 `training.save_interval` 配置 checkpoint 保存目录和间隔。当前 `MiniMaxM3Converter` 暂不支持 `dcp_to_hf`，训练 checkpoint 请使用 DCP 格式保存和续训。

---
<a id="jump3"></a>

## 数据集准备及处理

- 使用真实数据集训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/building_data_for_VLModel.md#real-data)。
- 图片和视频分别使用 `<image>` 和 `<video>` 占位符，并与 `images`、`videos` 字段一一对应；相对路径以 `dataset_dir` 为根目录解析。

## 微调

<a id="jump4.1"></a>

### 1. 准备工作

配置脚本前需完成**环境安装**、**权重下载及转换**和**数据集准备及处理**。

<a id="jump4.2"></a>

### 2. 配置参数

【数据目录配置】

根据实际情况修改 `minimax_m3_config.yaml` 中的 `model_name_or_path`、`dataset_dir`、`dataset` 和 `cache_dir`。多机训练时，不同节点不要同时写入同一个数据缓存目录。

【模块冻结配置】

在 `model.freeze` 中配置需要冻结的模块。示例默认冻结 `model.vision_tower`；如需全参数微调，请删除该项。

【模型保存加载配置】

在 `training` 中配置 `load`、`load_format`、`save` 和 `save_interval`。MiniMax-M3 模型规模较大，建议启用 `init_model_with_meta_device` 并加载 DCP 权重。

【EP并行配置】

通过 `parallel.expert_parallel_size` 设置 EP 并行度，并在 `parallel.ep_plan.dispatcher` 中选择 `alltoall`、`allgather` 或 `mc2`。示例保留 M3 的 NPU MoE 融合路径：

```yaml
parallel:
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts
    dispatcher: alltoall
    use_npu_fused_ops: true
```

【Attention配置】

`model.attn_implementation` 支持 `eager`、`sdpa`、`flash_attention_2` 和 `minimax_m3_msa`。NPU 训练默认使用 `flash_attention_2`；`minimax_m3_msa` 仅替换文本稀疏层，视觉层和文本 dense 层仍使用 Flash Attention。

【MoE aux loss配置】

通过 `features.loss_cfg.router_aux_loss_coef` 配置 router 辅助损失系数，默认值为 `0.0`，表示不计算该损失。

【性能优化配置】

- 重计算：通过 `features.recompute` 开启，作用范围由 `features.recompute_plan.apply_modules` 指定。
- [Chunk Loss](../../docs/zh/features/chunkloss.md)：通过 `features.enable_chunk_loss` 开启，降低长序列 LM Head loss 的显存峰值。
- [Activation Offload](../../docs/zh/features/async_activation_offload.md)：通过 `features.enable_activation_offload` 开启，将 checkpoint 激活异步卸载到 Host。
- Op Replay：在非重入式重计算中缓存指定算子的前向输出，减少反向阶段的重复计算。

8K 优化配置 `minimax_m3_vl_fsdp2_perf_8card.yaml` 只缓存 `npu.npu_fusion_attention.default`，并设置 `capacity_mb: 256`。不要直接在整个 `self_attn` scope 中加入 `aten.matmul.default`，因为 M3 Sparse Indexer 的 FP32 score 张量随序列长度平方增长，会引入较大的 Host/Device 传输开销。

【单机运行配置】

单卡调试时，将 `expert_parallel_size` 和 `NPUS_PER_NODE` 设置为 `1`。单机多卡默认使用 8 个进程，可通过环境变量覆盖。

【多机运行配置】

多机训练时，各节点使用相同配置，并分别设置 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK` 和 `NPUS_PER_NODE`。

---
<a id="jump4.3"></a>

### 3. 启动微调

修改 `minimax_m3_config.yaml` 中的模型、权重和数据路径后执行：

```bash
CONFIG_PATH=examples/minimax_m3_vl/minimax_m3_config.yaml \
NPUS_PER_NODE=8 \
bash examples/minimax_m3_vl/finetune_minimax_m3.sh
```

<a id="jump10"></a>

## 环境变量声明

| 环境变量 | 描述 | 取值说明 |
| --- | --- | --- |
| `TASK_QUEUE_ENABLE` | task queue 算子下发队列优化 | `0`：关闭；`1`：开启 |
| `CPU_AFFINITY_CONF` | CPU 亲和性配置 | `0`：关闭；`1`：开启粗粒度绑核；`2`：开启细粒度绑核 |
| `HCCL_CONNECT_TIMEOUT` | HCCL 建链超时时间 | 默认 `1200` 秒 |
| `PYTORCH_NPU_ALLOC_CONF` | NPU 缓存分配器配置 | 默认 `expandable_segments:True` |
| `MULTI_STREAM_MEMORY_REUSE` | 多流内存复用等级 | 默认 `2` |
| `MM_FORCE_EP_BALANCE` | 调试时强制均衡 EP 路由 | 默认 `0`；正式训练必须保持关闭 |
| `ASCEND_RT_VISIBLE_DEVICES` | 限制训练进程可见的 NPU | 按实际设备编号设置 |
| `NPUS_PER_NODE` | 单节点训练进程数 | 默认 `8` |

---
<a id="jump11"></a>

## 注意事项

1. `tensor_parallel_size` 当前必须设置为 `1`；示例暂未启用 Ring Attention 和 Ulysses CP。
2. `training.plugin` 中的 MiniMax-M3-VL 模型插件和 Hugging Face 数据集插件不可删除。
3. `MM_FORCE_EP_BALANCE=1` 会覆盖真实 router 输出，仅用于通信调试，不可用于正式训练。
4. `capacity_mb` 是 Op Replay 和 activation offload 共享的 HBM swap 预算；序列长度、micro batch size、并行策略或算子白名单变化后需重新调整。
