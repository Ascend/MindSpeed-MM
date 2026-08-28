# MindSpeed-MM 权重加载与保存使用指南

本文介绍 MindSpeed-MM FSDP2 后端使用的 Hugging Face（以下简称 HF）与 PyTorch Distributed Checkpoint（以下简称 DCP）权重格式的适配场景，权重加载与保存的配置流程，以及原始权重和模型布局不一致时的适配流程。

## 适配场景

### HF 权重

HF 权重通常以 `model.safetensors` 或多个 safetensors 分片保存，并配套提供 `config.json`、权重索引、tokenizer 和 processor 等模型资源文件。HF 格式具有良好的通用性，适合以下场景：

- 从 Hugging Face 或 ModelScope 获取预训练模型；
- 将训练后的模型导出进行推理、评测或发布；

### DCP 权重

DCP 是 PyTorch 原生的分布式检查点格式，可以按 FSDP2 分片保存和加载模型权重，并支持保存训练迭代信息、优化器状态和随机数状态，适用于以下场景：

- 大模型使用 meta device 初始化后按分片加载，降低初始化阶段的内存占用；
- FSDP2 分布式训练中保存完整训练状态，可以恢复数据迭代位置、优化器和随机数状态，从而进行断点续训；

## 配置方法

训练阶段可通过模型 YAML 配置文件中的 `training.load_format` 和 `training.save_format` 参数选择权重加载与保存的方式，当前支持 `hf`、`dcp` 和 `auto`。权重加载格式配置为 `auto` 时，会自动根据加载路径中的权重文件类型判断选择 `hf` 或 `dcp` 格式。权重保存格式配置为 `auto` 时，权重保存类型与加载类型保持一致；在断点续训场景（即保存优化器和随机数状态时），权重保存格式统一调整为 DCP 格式。

### HF 权重加载与保存

`training.load_format` 和 `training.save_format` 设置为 `hf` 时，框架会通过在线转换流程完成 HF 与 DCP 权重格式之间的转换。如果模型权重布局与 HF 原始权重布局存在差异，在线转换还会通过特定的权重转换流水线 `WeightTransformPipeline` 完成布局转换，具体配置方法如下：

```yaml
model:
  model_id: qwen3_5_moe  # 在线转换过程中会根据 model_id 选择相应权重转换流水线

training:
  # HF 权重加载
  load: /hf_load_path
  load_format: hf
  no_load_optim: true  # 加载 HF 权重时无法加载优化器状态
  no_load_rng: true  # 加载 HF 权重时无法加载随机数状态

  # HF 权重保存
  save: ./hf_save_path
  save_interval: 1000
  save_format: hf
  no_save_optim: true  # 保存 HF 权重时无法保存优化器状态
  no_save_rng: true  # 保存 HF 权重时无法保存随机数状态
```

### DCP 权重加载与保存

`training.load_format` 和 `training.save_format` 设置为 `dcp` 时，需要先将 HF 权重离线转换为 DCP 格式，再从 DCP 分片加载。如果模型权重布局与 HF 原始权重布局存在差异，离线转换还会复用权重转换流水线完成布局转换。

以 Qwen3.5 MoE 为例，先执行 HF → DCP 离线转换：

```bash
python -m mindspeed_mm.fsdp.tasks.checkpoint.converter hf_to_dcp \
    --model_id qwen3_5_moe \
    --hf_dir /path/to/hf_model \
    --dcp_dir /path/to/dcp_model \
    --num_workers 0

# 其中：
# model_id: 模型名称，用于选择权重转换流水线
# hf_dir: HF 权重路径
# dcp_dir: 转换后 DCP 权重的保存路径
# num_workers: 并行工作线程数，0表示串行执行，若存储I/O性能允许，可适当调大并发数以提升转换效率，推荐设置为 4
```

转换完成后，将 `training.load` 指向转换后的 DCP 权重根目录，并配置相关参数：

```yaml
training:
  # DCP 权重加载
  load: ./dcp_load_path
  load_format: dcp
  no_load_optim: true  # 断点续训时设置为 false 以加载优化器状态
  no_load_rng: true  # 断点续训时设置为 false 以加载随机数状态

  # DCP 权重保存
  save: ./dcp_save_path
  save_interval: 1000
  save_format: dcp
  no_save_optim: true  # 断点续训时设置为 false 以保存优化器状态
  no_save_rng: true  # 断点续训时设置为 false 以保存随机数状态
```

DCP 权重保存后，可通过以下命令转换回 HF 权重：

```bash
python -m mindspeed_mm.fsdp.tasks.checkpoint.converter dcp_to_hf \
    --model_id qwen3_5_moe \
    --dcp_dir /path/to/dcp_model/iter_000xx \
    --hf_dir /path/to/hf_model \
    --origin_hf_dir /path/to/origin_hf_model \
    --to_bf16 \
    --keep_origin_mtp_weights \
    --num_workers 0

# 其中：
# model_id: 模型名称，用于选择权重转换流水线
# dcp_dir: 保存的 DCP 格式权重路径，`iter_000xx` 表示保存的第 xx 步的权重
# hf_dir: 转换后 HF 格式的权重保存路径
# origin_hf_dir：原始 HF 格式权重路径
# to_bf16：开启时将权重数据类型从 FP32 精度转换为 BF16 精度
# keep_origin_mtp_weights：开启时若 DCP 不包含 MTP 权重则保留原始 HF 权重中的 MTP 相关权重
# num_workers: 并行工作线程数，0 表示串行执行，若存储 I/O 性能允许，可适当调大并发数以提升转换效率，推荐设置为 4
```

### 参数说明

以下说明 `training` 配置段中与权重加载、保存相关参数的含义：

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `load` | str \| null | `null` | 待加载的权重路径。 |
| `load_format` | str | `auto` | 加载权重类型，可选 `hf`、`dcp` 或 `auto`，设置为 `auto` 时根据 `load` 路径中文件类型自动识别权重类型。 |
| `no_load_optim` | bool | `false` | 是否跳过优化器状态加载，加载 DCP 权重进行断点续训时设置为 `false`，加载 HF 权重时须设置为 `true`。 |
| `no_load_rng` | bool | `false` | 是否跳过随机数状态加载，加载 DCP 权重进行断点续训时设置为 `false`，加载 HF 权重时须设置为 `true`。 |
| `save` | str \| null | `null` | 权重保存路径。 |
| `save_interval` | int | `1` | 保存权重的训练迭代间隔。 |
| `save_format` | str | `auto` | 保存权重类型，可选 `hf`、`dcp` 或 `auto`，设置为 `auto` 时保存权重类型与加载权重类型保持一致，开启断点续训时统一调整为 `dcp` 格式。 |
| `no_save_optim` | bool | `false` | 是否跳过优化器状态保存，保存 DCP 权重进行断点续训时设置为 `false`，关闭时权重保存类型自动调整为 DCP 格式。 |
| `no_save_rng` | bool | `false` | 是否跳过随机数状态保存，保存 DCP 权重进行断点续训时设置为 `false`，关闭时权重保存类型自动调整为 DCP 格式。 |

## 适配范围

当 HF 权重和模型权重布局不一致时，HF 权重在线加载和 DCP 权重离线转换需要通过转换流水线完成布局转换，当前已适配的模型列表如下：

| 后端 | 模型 | 转换内容 |
| :--- | :--- | :--- |
| FSDP2 | Qwen3.5 MoE | MoE 专家权重维度置换；MTP 专家权重合并与拆分。 |

如果 HF 权重 key、tensor 形状和模型内部参数完全一致，可以直接执行格式转换，无需新增转换规则。如需新增转换流水线，可按照下文适配流程进行添加。

## 转换流水线适配

权重转换流水线 `WeightTransformPipeline` 是 HF 权重与 FSDP2 模型内部权重之间的适配层。训练时在线加载 HF 权重或 HF 转 DCP 离线转换时会调用 `hf_to_dcp` 方法；在线保存 HF 权重或 DCP 转 HF 离线转换时会调用 `dcp_to_hf` 方法。

```mermaid
flowchart LR
    HF[HF safetensors] -->|hf_to_dcp| MODEL[FSDP2 模型内部权重]
    MODEL -->|dcp_to_hf| HF
```

### 转换流水线实现

在 `mindspeed_mm/fsdp/checkpoint/convert.py` 中新增模型权重转换流水线，并继承 `WeightTransformPipeline`：

```python
class NewModelWeightTransformPipeline(WeightTransformPipeline):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def hf_to_dcp(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> Optional[Tuple[str, torch.Tensor]]:
        # 将 HF key/tensor 转换为模型内部的 key/tensor
        return key, tensor

    def dcp_to_hf(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 将模型内部权重恢复为一个或多个 HF 权重
        return {key: tensor}
```

`hf_to_dcp` 接收单个 HF key/tensor，返回转换后的目标 key/tensor；如果权重需要跳过或等待其他 tensor 合并，则返回 `None`。`dcp_to_hf` 接收模型内部权重并返回 `{hf_key: tensor}` 字典。

当前可复用的转换函数如下：

| 转换函数 | 转换类型 | 说明 |
| :--- | :--- | :--- |
| `permute_moe_expert` | MoE 专家权重维度置换 | 对匹配正则表达式的权重执行维度置换，未匹配时保持不变。 |
| `reshape_fused_linear` | 融合 Linear 权重展平 | 将匹配权重的 `[E, input_dim, output_dim]` reshape 为 `[E × input_dim, output_dim]`。 |
| `rename_key` | 权重 key 前缀重命名 | 将 HF key 前缀替换为模型内部前缀；HF 前缀为空时，在原 key 前添加内部前缀。 |
| `merge_moe_expert_weights` | MoE 专家权重合并 | 将各专家独立的 `gate_proj` 和 `up_proj` 合并为融合 `gate_up_proj`，并合并各专家的 `down_proj`。 |
| `split_moe_expert_weights` | MoE 专家权重拆分 | 将融合专家权重恢复为各专家的 HF 权重。 |

### 转换流水线注册

完成实现后，在 `mindspeed_mm/fsdp/checkpoint/convert.py` 的 `WEIGHT_TRANSFORM_PIPELINES` 字典中添加模型标识与转换流水线的映射：

```python
WEIGHT_TRANSFORM_PIPELINES = {
    "model_id": NewModelWeightTransformPipeline,
}
```

字典中的 key 必须与 YAML 配置中的 `model.model_id` 一致。框架会根据该字段选择在线加载和保存使用的转换流水线。

## 注意事项

- 参数规模较小、无需布局转换或转换逻辑简单时推荐使用 HF 在线权重加载；大尺寸模型或转换逻辑复杂时推荐先将原始 HF 权重离线转换为 DCP 权重，避免每次启动进行大量 HF 权重转换导致启动耗时增加；
- HF 保存格式仅支持保存模型权重，不支持保存优化器状态和随机数状态，若需要进行断点续训，请保存为 DCP 格式；
