# 训练阶段评估验证功能

## 问题分析

在长时间训练过程中，仅观察训练集 loss 难以及时判断模型在未见数据上的表现，可能出现过拟合但无法及时发现的情况。为了在训练过程中周期性观察验证集 loss，需要在训练流程中支持验证集配置、验证频率控制以及验证阶段的模型状态切换。

## 解决方案

MindSpeed-MM 支持在训练过程中周期性执行 validation。训练达到指定间隔后，框架会暂停训练分支，将模型切换到 `eval()` 状态，遍历验证集计算验证 loss，并在验证结束后恢复训练状态继续训练。

## 使用方法

FSDP2 本次新增训练中验证能力；MCORE（Megatron）训练链路已有 validation 能力，本文一并说明两种后端的使用方式。两者配置入口不同：FSDP2 使用 `val_dataset_param`、`val_interval` 等新增字段；MCORE 使用 Megatron 原有的 `eval_interval`、`eval_iters` 以及数据集中的验证集字段。

### FSDP2（native FSDP2）

在模型 YAML 配置文件中增加验证集配置，并设置验证间隔：

```yaml
data:
  dataset_param: &TRAIN_DATASET_PARAM
    dataset_type: huggingface
    basic_parameters: &TRAIN_BASIC_PARAMETERS
      dataset_dir: ./data
      dataset: ./data/train.json
      cache_dir: ./cache_dir/train

  val_dataset_param:
    <<: *TRAIN_DATASET_PARAM
    basic_parameters:
      <<: *TRAIN_BASIC_PARAMETERS
      dataset: ./data/val.json
      cache_dir: ./cache_dir/val

training:
  micro_batch_size: 4
  val_interval: 1000
  val_micro_batch_size: 1
```

- `data.val_dataset_param`：验证集配置，结构与 `data.dataset_param` 一致，可通过 YAML 锚点复用训练集配置，仅覆盖验证集路径、缓存路径等差异字段。
- `training.val_interval`：验证间隔，单位为 step。默认 `0`，表示关闭验证；大于 `0` 时开启验证。
- `training.val_micro_batch_size`：验证阶段的 micro batch size。未设置时默认复用 `training.micro_batch_size`。

验证开启后，如果未配置验证集，训练启动阶段会报错。验证 dataloader 会关闭 shuffle，并使用独立 seed，避免影响训练 dataloader 的采样顺序。

可参考 `examples/qwen3_5/qwen3_5_35B_config.yaml`。

当前 FSDP2 部分模型可参考 Qwen3.5 的实现方式，其他模型支持情况以具体实现为准。

### MCORE（Megatron）

MCORE 训练链路使用 Megatron 原有评估参数控制验证频率和验证步数。以 shell 脚本方式启动时，相关参数通常在 `OUTPUT_ARGS` 中设置：

```shell
OUTPUT_ARGS="
    ...
    --eval-interval 1000 \
    --eval-iters 10 \
    ...
"
```

- `--eval-interval`：验证间隔，单位为 iteration。
- `--eval-iters`：每次验证执行的迭代次数。

验证集可以在数据配置的 `basic_parameters` 中通过 `val_dataset` 显式配置：

```json
{
  "dataset_param": {
    "basic_parameters": {
      "dataset": "./data/train.json",
      "val_dataset": "./data/val.json",
      "val_max_samples": null
    }
  }
}
```

- `val_dataset`：显式指定验证集路径。
- `val_max_samples`：限制验证集样本数，常用于调试。

MCORE（Megatron）侧 validation 能力依赖具体模型训练链路实现，部分模型可参考 Qwen2.5VL 的实现方式。其他模型支持情况以具体实现为准。

### 约束说明

- 当前功能仅支持训练过程中的 loss 验证，要求模型 forward 返回 loss，且验证集 batch 格式与训练集一致。
- 当前不支持生成式评测和任务指标评估。
