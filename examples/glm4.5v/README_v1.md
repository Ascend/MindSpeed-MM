# Glm4.5v 使用指南（FSDP2 后端）

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [权重下载及转换](#权重下载及转换)
  - [权重下载](#1-权重下载)
  - [权重转换](#2-权重转换)
- [数据集准备及处理](#数据集准备及处理)
- [微调](#微调)
  - [准备工作](#1-准备工作)
  - [配置参数](#2-配置参数)
  - [启动微调](#3-启动微调)
  - [HF权重转换](#4-hf权重转换)
- [环境变量声明](#环境变量声明)
- [跨后端对拍](#跨后端对拍)
- [注意事项](#注意事项)

## 版本说明

本指南使用纯 FSDP2 后端，训练入口为 `mindspeed_mm/fsdp/train/trainer.py`，配置文件为 `examples/glm4.5v/glm4_5v_106B_config_v1.yaml`，启动脚本为 `examples/glm4.5v/finetune_glm4_5v_106B_v1.sh`。Megatron 后端请参考 [README.md](README.md)。

### 参考实现

```shell
url=https://github.com/huggingface/transformers.git
commit_id=8cb5963
```

### 变更记录

2026.08.29：新增纯 FSDP2 后端。

2026.10.11：增加 NPU EP=1 grouped-MoE 路径及前向/梯度测试，补充 128 专家对拍结果。

---

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/guides/installation/install_guide.md)，完成昇腾软件安装。

<a id="jump1.2"></a>

### 2. 环境搭建

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
mkdir -p logs data ckpt

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout d76dbddd4517d48a2fc1cd494de8b9a6cfdbfbab

# 安装mindspeed及依赖
pip install -e .
cd ..
# 安装mindspeed mm及依赖
pip install -e .

# 安装新版transformers（支持glm4.5v模型）
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 8cb5963
pip install -e .
cd ..
```

后续命令均在 `MindSpeed-MM` 仓库根目录执行。纯 FSDP2 启动脚本会设置 `NON_MEGATRON=true`；直接调用训练入口或权重转换命令时也需要设置该变量。

---

## 权重下载及转换

### 1. 权重下载

从 Hugging Face 下载 [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V)，将模型权重保存到 `ckpt/hf_path/GLM-4.5V`。

> [!NOTE]
>
> 如无法访问 Hugging Face，可从 ModelScope 下载，并核对模型及文件版本。

### 2. 权重转换

本后端使用三维堆叠专家权重，转换结果应与 Megatron 后端分目录保存。

设置 `GLM4V_DCP_3D=1` 保留三维专家权重，使用空 `dcp_prefix` 保持与模型一致的权重键名：

```bash
HF_DIR=./ckpt/hf_path/GLM-4.5V
MM_DIR=./ckpt/mm_path/GLM-4.5V-fsdp2

NON_MEGATRON=true GLM4V_DCP_3D=1 mm-convert ExpertMergeDcpConverter hf_to_dcp \
    --hf_dir "${HF_DIR}" \
    --save_dir "${MM_DIR}" \
    --dcp_prefix "" &&

# 转换成功后复制模型配置、tokenizer 和 processor 文件
for f in config.json generation_config.json chat_template.jinja tokenizer.json \
         tokenizer_config.json preprocessor_config.json video_preprocessor_config.json; do
    if [ -f "${HF_DIR}/${f}" ]; then
        cp "${HF_DIR}/${f}" "${MM_DIR}/"
    fi
done
```

转换完成后，将 `glm4_5v_106B_config_v1.yaml` 中的 `model.model_name_or_path` 和 `training.load` 均指向 `MM_DIR`。保留默认的 `training.init_model_with_meta_device: true` 和 `training.load_strict: true`。

---

## 数据集准备及处理

- 使用**真实数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/features/data/building_data_for_VLModel.md#real-data)（下载COCO2017 → 下载LLaVA-Instruct-150K标注 → 运行转换脚本生成`mllm_format_llava_instruct_data.json`）。
- 使用**虚构数据**做功能/性能测试：参考[针对VL模型的数据构造 · 使用虚构数据](../../docs/zh/features/data/building_data_for_VLModel.md#mock-data)。

## 微调

### 1. 准备工作

启动前完成环境安装、权重转换和数据准备，并根据实际路径修改配置文件。

### 2. 配置参数

配置文件为 `examples/glm4.5v/glm4_5v_106B_config_v1.yaml`。

【数据目录配置】

`data.dataset_param.preprocess_parameters.model_name_or_path` 指向原始 Hugging Face 目录，用于加载 processor 和 tokenizer。`dataset` 指向标注 JSON，标注中的媒体相对路径以 `dataset_dir` 为基准；多机不要共用同一个缓存写入目录。

```yaml
data:
  dataset_param:
    preprocess_parameters:
      model_name_or_path: ./ckpt/hf_path/GLM-4.5V
    basic_parameters:
      dataset_dir: ./data
      dataset: ./data/mllm_format_llava_instruct_data.json
      cache_dir: ./data/cache_dir
```

【模型保存加载配置】

将模型配置目录和权重加载目录同时设为本后端转换后的目录：

```yaml
model:
  model_name_or_path: ./ckpt/mm_path/GLM-4.5V-fsdp2
training:
  init_model_with_meta_device: true
  load: ./ckpt/mm_path/GLM-4.5V-fsdp2
  load_format: dcp
  load_strict: true
  no_load_optim: true
  no_load_rng: true
  save: ./save_dir
  save_interval: 10000
```

`training.save` 为空时不会保存检查点。初次加载转换权重时不加载优化器和随机数状态；从完整训练检查点续训时，应将 `training.load` 改为训练检查点目录，并按检查点内容调整 `training.no_load_optim` 和 `training.no_load_rng`。

【并行与优化特性配置】

| 功能 | 配置项 | 默认值 |
| --- | --- | --- |
| FSDP2 | `parallel.fully_shard_parallel_size` | `auto` |
| EP=1 NPU grouped MoE | `model.use_grouped_expert_matmul` | `true` |
| 专家并行 | `parallel.expert_parallel_size` | `16` |
| NPU MoE 融合 | `parallel.ep_plan.use_npu_fused_ops` | 框架默认开启 |
| EP 动态负载均衡 | `features.enable_ep_balance` | `false` |
| Ring CP | `parallel.ring_attention_size` | `1` |
| Ulysses CP | `parallel.ulysses_parallel_size` | `1` |
| 静态分块 loss | `features.enable_chunk_loss` | `true` |
| 动态分块 loss | `features.enable_dynamic_chunk_loss` | `false` |
| 重计算 | `features.recompute` | `true` |
| 异步激活卸载 | `features.enable_activation_offload` | `false` |
| 确定性计算 | `training.use_deter_comp` | `false` |

上下文并行默认关闭。启用时，将 `parallel.ring_attention_size` 或 `parallel.ulysses_parallel_size` 设为所需规模，并保证它们的乘积与总进程数及其他并行维度兼容；`model.attn_implementation` 保持为 `flash_attention_2`。Ring CP 依赖 MindSpeed 提供的 Ring Attention 算子。

基础 EP 默认使用 `alltoall` dispatcher。GLM-4.5V 使用带 `e_score_correction_bias` 的 aux-free 路由，`features.loss_cfg.router_aux_loss_coef` 保持为 0。开启 `features.enable_ep_balance` 时仅支持 `alltoall`，并会复制热点专家，需预留额外显存。

静态和动态分块 loss 不能同时启用。改用动态分块时配置：

```yaml
features:
  enable_chunk_loss: false
  enable_dynamic_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    total_chunk_size: 4096
```

重计算与异步激活卸载的 `apply_modules` 均覆盖 `model.visual.blocks.{*}` 和 `model.language_model.layers.{*}`。重计算默认开启；异步激活卸载默认关闭，启用时将 `features.enable_activation_offload` 设为 `true`，并确认主机内存和 PCIe 带宽满足要求。各特性的显存与性能收益取决于模型规模、序列长度和并行配置。

### 3. 启动微调

根据实际安装位置修改 `finetune_glm4_5v_106B_v1.sh` 中的 CANN 环境脚本路径。完整 106B 模型采用多机配置；所有节点使用同一个主节点 IP、端口和节点数，`NODE_RANK` 各不相同：

```bash
export NPUS_PER_NODE=16
export MASTER_ADDR="<rank-0-node-ip>"
export MASTER_PORT=6000
export NNODES=8
export NODE_RANK=0  # 各节点分别设置为 0 至 7

bash examples/glm4.5v/finetune_glm4_5v_106B_v1.sh
```

启动脚本固定读取 `examples/glm4.5v/glm4_5v_106B_config_v1.yaml`。单卡功能调试时，先准备减层模型对应的 HF config 和权重，按“权重转换”中的纯 FSDP2 命令转换到单独目录，再复制配置文件：

```bash
cp examples/glm4.5v/glm4_5v_106B_config_v1.yaml examples/glm4.5v/glm4_5v_debug.yaml
```

修改 `glm4_5v_debug.yaml`：数据预处理路径指向减层 HF 目录，`model.model_name_or_path` 和 `training.load` 指向对应的转换目录；将 `parallel.expert_parallel_size`、`parallel.fully_shard_parallel_size`、`parallel.ring_attention_size` 和 `parallel.ulysses_parallel_size` 均设为 `1`，并按需缩短 `training.train_iters`。完成修改后，直接指定该 YAML 启动：

```bash
# 根据实际情况修改 CANN 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NON_MEGATRON=true torchrun --standalone --nproc_per_node=1 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/glm4.5v/glm4_5v_debug.yaml
```

### 4. HF权重转换

训练完成后，将 `save_dir` 中的检查点转换成 Hugging Face 格式：

```bash
NON_MEGATRON=true GLM4V_DCP_3D=1 mm-convert ExpertMergeDcpConverter dcp_to_hf \
    --hf_dir ./ckpt/hf_path/GLM-4.5V \
    --dcp_dir ./save_dir/iter_000xxxx \
    --save_dir ./ckpt/hf_export/GLM-4.5V \
    --dcp_prefix "" \
    --hf_prefix ""
```

`--hf_dir` 指向原始 Hugging Face 权重，`--dcp_dir` 指向要导出的训练检查点，`--save_dir` 指向导出目录。将 `iter_000xxxx` 替换为实际保存步数对应的目录名，转换完成后可使用 Hugging Face 模型进行推理。

## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                           | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                   |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志 |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景 | `0`: 关闭<br>`1`: 开启                                                                           |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步         | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                    |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                                  | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                    |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为    | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |
| `NPU_ASD_ENABLE`   | 控制是否开启TorchNPU的特征值检测功能 | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`   | 控制算子执行时是否启动同步模式 | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                               |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                                  | 整数值（如 `1`, `8` 等）                                                                            |
| `NON_MEGATRON` | 选择纯 FSDP2 路径，跳过 Megatron 补丁导入 | 纯 FSDP2 设为 `true` |
| `GLM4V_DCP_3D` | 控制 GLM-4.5V 专家权重转换布局 | 纯 FSDP2 转换设为 `1`；Megatron 转换不设置 |

---

## 跨后端对拍

2026-10-11 在 8 张 Ascend 910B3 上完成 2000 步验证：文本和视觉各保留 4 层，保留 128 个 routed experts 和 top-k 8，EP=1、CP=1、FSDP=8，使用固定的 2048 条真实 LLaVA/COCO 图文样本。该配置不是完整 106B 模型，也不替代 EP>1 的验证。

| 指标 | 结果 |
| --- | --- |
| 2000 步 loss 平均相对误差 | 1.617965%，低于 2% |
| 最大单步相对误差 | 14.324236%，不代表每步都小于 2% |
| Megatron / pure FSDP2 平均耗时 | 922.725 / 903.014 ms/step（排除前 200 步） |

平均相对误差以 Megatron loss 为分母，对全部 2000 个相同步号计算，不丢弃尖峰或后段。本轮只验证一个 seed；更换模型规模、并行配置或数据后需重新对拍。

该结果使用专门对齐的配置，**不是直接运行上面的默认训练 YAML 所得**：

- seed=1234、micro batch=1、global batch=8、shuffle=false；两端从对应的初始 DCP 加载，不恢复 optimizer/RNG，保持同一 2000 步 cosine 学习率计划、warmup ratio=0.1、lr=1e-5、weight decay=0。
- loss 使用 default，关闭 ChunkLoss；attention 按 Megatron 的实际运行路径设为 SDPA。
- 只冻结 `model.visual.patch_embed`、`model.visual.blocks`、`model.visual.merger`；视觉与文本层均启用 reentrant recompute。
- Megatron 的 BF16 包装会同时转换浮点 buffer，而纯 FSDP2 的参数混合精度不会转换 buffer。本轮在独立验证入口中、模型准备后且检查点加载前，将浮点 buffer（含 RoPE `inv_freq`）对齐为 BF16，保留 FP32 参数，并同步 RoPE 的 `original_inv_freq` 引用。

buffer 对齐用于复现既有 Megatron 数值语义，不表示 BF16 RoPE 比 FP32 更准确；本 PR 未把该策略默认应用到通用 Trainer。对齐后首步捕获的 42 个张量字段与基线逐元素一致，但反向和后续训练仍存在数值差异。完整配置、验证入口和曲线见 [PR !2875](https://gitcode.com/Ascend/MindSpeed-MM/merge_requests/2875)；临时诊断脚本和训练日志不纳入模型代码目录。

## 注意事项

- 完整 106B 模型需要多机运行；单卡示例用于已经准备好减层 config 和匹配权重的功能调试。
- 两种后端的专家权重布局不同，转换结果和训练检查点不能混用。纯 FSDP2 导入与导出均需保留 `GLM4V_DCP_3D=1` 和空前缀参数。
- CP、EP 需要相应的多卡环境。单卡调试通过不能代替多卡通信与并行配置验证。
