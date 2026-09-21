# Preprocess On Fly

## 背景与挑战

非流式数据加载场景下，数据预处理默认通过 `dataset.map` 在启动阶段全量执行，并将结果落盘到 Arrow 缓存。这种方案在多模态训练场景下存在以下瓶颈：

- 启动慢：针对大规模数据集全量预处理耗时高，期间可能因数据处理过久导致超时报错退出
- 磁盘占用高：多模态数据（图像、视频）单条样本预处理后全量落盘占用大量磁盘空间

针对上述挑战，可以使用在线预处理（Preprocess On Fly）策略。

## 解决方案

- 按需执行：将 `preprocess_func` 通过 `set_transform` 挂载为 dataset 的 transform，在训练过程中每次读取 batch 时才触发预处理，不落盘
- 并行预取：配合 DataLoader 的 `num_workers` 多进程预取机制，掩盖预处理耗时

## 使用方法

**该特性仅在非流式数据加载场景下针对huggingface数据集类型生效**，通过 `preprocess_on_fly` 参数控制，当前在Kimi-K2.5及Qwen3.5系列模型中已默认开启。使用示例如下：

```yaml
basic_parameters:
  streaming: false  # 需关闭流式加载
  preprocess_on_fly: true

dataloader_param:
  num_workers: 8  # 通过多进程预取掩盖预处理耗时
```

### 参数详解

- `streaming`：流式加载开关，`preprocess_on_fly` 仅在 `streaming: false` 时生效，该参数默认关闭
- `preprocess_on_fly`：是否在训练时进行预处理，默认 `false`。开启后走 `set_transform` 路径，不调用 `dataset.map`，`preprocessing_batch_size` 和 `preprocessing_num_workers` 不生效
- `num_workers`：DataLoader 的 worker 进程数，通过多进程预取掩盖预处理耗时。`num_workers=0` 时预处理在主进程同步执行，会阻塞训练
