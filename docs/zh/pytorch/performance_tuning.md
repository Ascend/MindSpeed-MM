# 性能调优

Last updated: 12/15/2025. Author: zhuyanru

性能优化是训练的重要环节，合理的性能调优可以显著提升模型训练效率，降低资源消耗。本文档介绍了 MindSpeed-MM 在性能调优方面的最佳实践和方法。首先，我们介绍性能数据采集的方法，然后介绍性能分析的流程，最后介绍常用的性能调优方法。

性能在本文中，指机器（GPU、NPU 或其他平台）在指定模型和输入数据的背景下，完成一次端到端训练所需要花费的时间，考虑到不同模型的训练数据量和训练轮次（epoch）差异，此处定义的性能是在完成一个 batch 训练所需花费的时间。而这里的端到端，我们通常是指完成一个人工智能模型单步训练的过程，也就是说，本文所讨论的性能的衡量和性能的优化，都是站在模型角度上。

对于一个 batch 而言，时间主要由以下部分构成：

单 batch 总时间 = 数据加载时间 + 模型前反向时间 + 优化器时间 + 模型后处理时间 + 通信时间 + 调度时间

各组成成分介绍如下：

- 数据加载时间：模型在加载自身所需要的数据（如图片、视频和文本等）的时间，包括将数据从硬件存储设备读取到 CPU（Central Processing Unit）中、CPU 中数据的预处理（编解码等操作）、CPU 数据放到 device 上的时间。
- 模型前向和反向时间：深度学习模型的前向过程和反向过程的时间，即 Forward 和 Backward 过程，包含前向的数据计算和反向的数据微分求导的时间。
- 优化器时间：模型参数更新时间。
- 模型后处理时间：优化器更新后的时间，包括数据的后处理或者一些必要的同步操作，通常取决于模型特有操作。
- 通信时间：单节点内卡之间和多节点间节点之间的通信时间。由于 PyTorch 的特殊机制，在通信和计算可以并行的情况下，表示未被计算掩盖的通信时间。
- 调度时间：模型从 CPU 的指令到调用 NPU 侧的核（Kernel）所需要的时间。

## 性能数据采集

MindSpeed-MM 提供了 Profiling 采集工具，支持静态采集和动态采集两种模式，用于采集模型训练过程中的性能数据。详细使用方法请参考 [Profiling 采集工具](../tools.md#profiling采集工具)。

## 性能分析流程

采集到性能数据后，可通过 [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md) 对性能数据进行可视化分析，定位性能瓶颈。

MindStudio Insight 是昇腾提供的性能分析工具，支持对 Profiling 采集的数据进行多维度分析，包括：

- 算子耗时分析：识别耗时较长的算子，定位计算瓶颈
- 通信耗时分析：分析通信与计算的时间占比，优化通信策略
- 内存分析：查看显存使用情况，识别显存瓶颈
- 流水线分析：分析流水线并行的空泡比例

## 性能调优方法

MindSpeed-MM 提供了多种性能调优特性，可根据实际场景选择合适的策略。各特性的详细使用方法请参考 [特性目录](../features/feature_list.md) 中的对应文档。

常用的性能调优特性包括：

- **序列并行**：通过切分序列维度降低单卡计算量，支持 Ulysses、RingAttention、USP 等算法，详见 [DiT USP](../features/dit_usp.md) 和 [DiT Ring Attention](../features/dit_ring_attention.md)
- **异步激活值卸载**：将激活值卸载至 Host 侧，利用异步机制使拷贝被计算掩盖，降低峰值显存，详见 [异步激活值卸载](../features/async_activation_offload.md)
- **异构并行**：支持不同维度的并行策略组合，详见 [异构并行](../features/hetero_parallel.md)
- **ChunkLoss**：对序列维度分块计算 loss，避免同时保留整个序列的 logits，降低显存峰值，详见 [ChunkLoss](../features/chunkloss.md)
- **Bucket 重排**：优化通信效率，详见 [Bucket 重排](../features/bucket_reordering.md)
