# 性能调优

性能优化是训练的重要环节，合理的性能调优可以显著提升模型训练效率，降低资源消耗。本文档介绍了MindSpeed-MM在性能调优方面的最佳实践和方法。首先，我们介绍性能数据采集的方法，然后介绍性能分析的流程，最后介绍常用的性能调优方法。

性能在本文中，指机器（GPU、NPU或其他平台）在指定模型和输入数据的背景下，完成一次端到端训练所需要花费的时间，考虑到不同模型的训练数据量和训练轮次（epoch）差异，此处定义的性能是在完成一个batch训练所需花费的时间。而这里的端到端，我们通常是指完成一个人工智能模型单步训练的过程，也就是说，本文所讨论的性能的衡量和性能的优化，都是站在模型角度上。

对于一个batch而言，时间主要由以下部分构成：

单batch总时间 = 数据加载时间 + 模型前反向时间 + 优化器时间 + 模型后处理时间 + 通信时间 + 调度时间

各组成成分介绍如下：

- 数据加载时间：模型在加载自身所需要的数据（如图片、视频和文本等）的时间，包括将数据从硬件存储设备读取到CPU（Central Processing Unit）中、CPU中数据的预处理（编解码等操作）、CPU数据放到device上的时间。
- 模型前向和反向时间：深度学习模型的前向过程和反向过程的时间，即Forward和Backward过程，包含前向的数据计算和反向的数据微分求导的时间。
- 优化器时间：模型参数更新时间。
- 模型后处理时间：优化器更新后的时间，包括数据的后处理或者一些必要的同步操作，通常取决于模型特有操作。
- 通信时间：单节点内卡之间和多节点间节点之间的通信时间。由于PyTorch的特殊机制，在通信和计算可以并行的情况下，表示未被计算掩盖的通信时间。
- 调度时间：模型从CPU的指令到调用NPU侧的核（Kernel）所需要的时间。

## 性能数据采集

MindSpeed-MM提供了Profiling采集工具，支持静态采集和动态采集两种模式，用于采集模型训练过程中的性能数据。两个训练后端的配置入口不同：

- **FSDP2后端**：在训练YAML的`tools`段配置，`tools.profile`用于性能数据采集（`enable`、`profile_type`、`ranks`等字段），`tools.memory_profile`用于显存快照采集（`enable`、`start_step`、`end_step`、`save_path`等字段），可参考各FSDP2模型样例YAML（如`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`）中的`tools`段。
- **MCORE（Megatron）后端**：通过`mindspeed_mm/tools/tools.json`配置采集开关与参数，详细使用方法请参考[Profiling采集工具](../tools.md#profiling采集工具)；

采集profiling时，建议同时记录以下关键上下文信息，便于后续分析：

- 模型名称、参数量、分布式策略（TP/PP/DP/EP/SP设置）
- 训练配置（batch size、序列长度、梯度累积步数）
- NPU型号、集群规模、CANN/TorchNPU插件版本号
- 是否开启重计算、激活值卸载等特性

## 性能分析流程

### 性能分析工具

采集到性能数据后，可通过[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)对性能数据进行可视化分析，定位性能瓶颈。

MindStudio Insight是昇腾提供的性能分析工具，支持对Profiling采集的数据进行多维度分析，包括：

- 算子耗时分析：识别耗时较长的算子，定位计算瓶颈
- 通信耗时分析：分析通信与计算的时间占比，优化通信策略
- 内存分析：查看显存使用情况，识别显存瓶颈
- 流水线分析：分析流水线并行的空泡比例

#### 多卡Timeline对齐

多卡采集时各卡timeline各自记录，多机场景下因各机器时钟不一致，可能导致各卡timeline相对位置不对齐，影响跨卡分析（如快慢卡、通信等待）的判断；这属于采集数据在分析时的展示问题，不影响实际训练。MindStudio Insight对单卡、集群、多模型场景已实现timeline相对位置自动对齐；若多机场景仍不准，可使用其单卡或泳道维度的时间校准功能（在时间线界面设置偏移量，将对应卡/泳道左右平移），手动对齐各卡时间基准，详见[MindStudio Insight系统调优指南](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/system_tuning.md)中“时间线（Timeline） - 支持单卡和泳道时间对齐”小节。

### 性能拆解分析方法

拿到profiling数据后，建议按照以下维度进行系统拆解：

1. **计算vs通信vs空闲占比**：首先看整体时间中计算、通信和Free的比例。Free占比高通常意味着Host bound或调度问题，此时可在MindStudio Insight工具timeline中打开**下发曲线**（Flow Events）：曲线陡峭或出现明显空隙的时间段，往往就是Host来不及下发、NPU空等的位置，对照这些位置即可定位下发瓶颈并做针对性优化（参见[调度下发优化](#调度下发优化)）；通信占比高需要检查通算掩盖情况。
2. **快慢卡排查**：对比各卡的耗时分布，如果存在某张卡通信或下发明显慢于其他卡，需排查是否为下发bound、负载不均衡或硬件问题。
3. **算子级拆解**：将计算时间按算子类型（Cube/Vector/AI_CPU）分解，识别占比最高的算子类型。Cube算子（MatMul/FA等）MFU低说明计算密度不足，Vector算子占比过高说明存在可融合空间，出现AI_CPU算子则意味着该算子回退到CPU执行、通常较慢，应设法消除或替换为NPU算子。
4. **子模块MFU分析**：对模型子模块（Attention、MLP等）分别计算MFU，精确定位计算瓶颈。

#### 通信耗时分析

分析通信时间在总时间中的占比，重点关注未被计算掩盖的部分，这是优化的主要目标。若通信未与计算充分重叠，可通过调整预取时机（提前发起AllGather）来改善通算掩盖。

#### 快慢卡排查

大规模集群训练时，快慢卡（部分卡明显慢于其他卡）是常见性能瓶颈。排查思路：(1)所有机器跑单机脚本对比性能数据；(2)从小集群逐步扩大，二分法定位慢节点；(3)采集集群profiling定位通信等待最长的卡。常见根因包括：下发bound（某张卡ACL接口耗时增长）、负载不均衡（多模态数据图像大小不一导致ViT部分差异大）、网络拥塞（跨Block组网或HCCL算法配置不当）、Host资源抢占（节点上非训练进程占用CPU）。

从通信timeline定位时，应先区分耗时是**真正的通信传输**还是**同步等待**：若AllGather / ReduceScatter等通信算子内存在较长的`notify_wait`，说明本卡已就绪、正在等待较慢的rank，根因多在慢卡（下发或显存重整产生异常Free）而非本卡，调整本卡的通信预取难以见效，应优先排查并消除慢卡上的下发或显存重整问题。

此外，**显存重整**也是导致长时间Free的常见原因。当显存占用过高时，分配器需要先把已有显存碎片整理、整合出一块连续空间才能完成分配，这一过程开销较大，会在timeline上表现为明显且较长时间的Free，显著拖慢训练。排查时建议在采集profile时一并开启memory采集（FSDP2下将YAML的`tools.memory_profile`段设`enable: true`），若观察到整体显存占用出现明显的上升下降（骤升骤降），通常即为触发了显存重整。由于重整对性能影响较大，应设法降低显存占用，例如降低序列长度或扩大卡数。

#### MFU分解分析

MFU（Model FLOPs Utilization，模型算力利用率）是指训练过程中实际达到的FLOPs与硬件理论峰值FLOPs之比，反映硬件算力被有效利用的程度，是衡量计算效率的常用指标。通过将MFU按模型子模块（如Attention、MLP等）和按阶段（前向、反向、重计算）进行分解，可以精确定位计算瓶颈。

分析步骤如下：

1. **计算理论FLOPs**：根据模型结构和输入shape，分别计算各模块的理论FLOPs。
2. **获取实际耗时**：从profiling数据中提取各模块在前向、反向各阶段的真实耗时。
3. **计算子模块MFU**：`子模块MFU = 子模块FLOPs / (子模块耗时 × 理论算力)`。
4. **对比定位瓶颈**：对比各子模块的MFU值，识别利用率低的模块。

实践中，不同子模块的MFU高低往往有其结构性原因：Attention层以稠密矩阵乘（如FlashAttention）为主，MFU通常较高；MoE experts层因包含permute/unpermute等非计算操作，MFU通常较低。解读分解结果时，关键是区分“结构性偏低”（如MoE routing本身就含大量非计算开销，难以改变）与“可优化的偏低”（如可融合的Vector算子、冗余的Cast、低效访存等）。后者才是优化对象。优化的目的是用合适的手段消除这些浪费、提升真实性能，而不是去抬高MFU这个数字本身。

## 性能调优方法

MindSpeed-MM提供了多种性能调优特性，可根据实际场景选择合适的策略。各特性的详细使用方法请参考[特性目录](../features/feature_list.md)中的对应文档。部分特性与训练后端绑定，使用前请先确认当前所用后端。

### 两后端通用特性

以下为FSDP2与MCORE均支持的框架特性：

#### 异步激活值卸载

将激活值卸载至Host侧，利用异步机制使拷贝被计算掩盖，降低峰值显存，详见[异步激活值卸载](../features/async_activation_offload.md)

#### ChunkLoss

对序列维度分块计算loss，避免同时保留整个序列的logits，降低显存峰值，详见[ChunkLoss](../features/chunkloss.md)

#### 重计算

在反向传播时重新计算前向的部分中间结果，以计算时间换取显存空间。适用于显存紧张、需要增大MBS或序列长度的场景。

#### 融合算子

将多个连续小算子融合为单个大算子（如RotaryMul、RmsNorm、SwiGlu等），减少kernel launch开销和显存带宽消耗。融合算子的使用由框架自动应用，无需用户手动配置。

#### CPU绑核

TorchNPU提供了`CPU_AFFINITY_CONF`环境变量，用于控制CPU端任务绑核，减少上下文切换和跨NUMA访问延迟：`0`或未设置表示不启用；`1`表示开启粗粒度绑核；`2`表示开启细粒度绑核。开启绑核后，还可选地为指定NPU卡自定义绑核区间，例如`export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47`。详见[环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/EnvVariables/docs/zh/environment_variable_reference/CPU_AFFINITY_CONF.md)。

#### 调度下发优化

当Host侧算子下发成为瓶颈（profiling中大段Free且CPU侧无任务执行）时，可尝试以下TorchNPU插件环境变量：

| 环境变量 | 作用 | 推荐值 |
|----------|------|--------|
| `TASK_QUEUE_ENABLE` | 控制算子下发队列优化等级 | `2`（开启Level 2优化） |
| `ACLNN_CACHE_LIMIT` | 算子缓存数目上限 | `100000` |
| `PYTORCH_NPU_ALLOC_CONF` | 控制缓存分配器行为 | `expandable_segments:True`（开启虚拟内存） |
| `ASCEND_LAUNCH_BLOCKING` | 是否强制算子同步下发（调试用） | 正常训练取`0`或不设置；需定位报错算子、获取精确堆栈时临时取`1`（会同步阻塞下发、拖慢训练，调试完应改回） |

#### Python GC问题

Python的垃圾回收（GC）机制可能导致训练性能抖动。当观察到周期性性能劣化且profiling中存在大段Free时，可以尝试调整GC阈值`gc.set_threshold(700, 10, 10000)`、在保存checkpoint后主动调用`gc.collect()`，极端场景下可尝试`gc.disable()`但需注意内存泄漏风险。

### FSDP2后端特性

以下方法仅适用于FSDP2插件式后端（入口为`mindspeed_mm/fsdp/train/trainer.py`）。

#### 显存优化

显存是训练吞吐的关键约束，释放显存可以增大数据规模（MBS或序列长度），从而实现更好的通算掩盖。以下方法均可用于降低显存峰值。

- **ChunkMBS**：结合重计算与异步激活卸载，对Batch维度进行细粒度切分（Chunk），使得一次参数Unshard后可以完成多个micro-chunk的前反向计算，减少通信次数、提升通算掩盖效果。在Qwen3.5 35B模型上实测整网收益约5%。详见[ChunkMBS](../features/chunkmbs.md)。使用ChunkMBS需同时开启重计算和异步激活值卸载，且`apply_modules`需被前两者覆盖。
- **async_activation_offload（异步激活值卸载）**：FSDP2通过`enable_activation_offload`开启，原理与详情见[异步激活值卸载](../features/async_activation_offload.md)。

#### Host内存优化

- **Meta Device初始化**：不开启时多rank会占用大量Host内存；开启后先在meta device上构建空结构，各rank只初始化自己那份分片，节省host内存。使用方式：在YAML配置中设置`init_model_with_meta_device: true`。

#### 通信与计算重叠

- **序列并行**：Ulysses上下文并行，通过训练YAML的`parallel.ulysses_parallel_size`配置，原理见[Unaligned Ulysses CP](../features/unaligned_ulysses_cp.md)
- **Prefetch配置**：通过提前发起参数AllGather实现通信与计算重叠，由`num_to_forward_prefetch`（前向预取层数，默认0）和`num_to_backward_prefetch`（反向预取层数，默认1）控制（均定义于`mindspeed_mm/fsdp/params/parallel_args.py`）。两点需注意：
  - 开启EP（Expert Parallelism）场景下，前向预取的AllGather可能与EP域的All2All抢占带宽，应检查预取层数是否合理。
  - 预取顺序由`fsdp_plan.apply_modules`的配置顺序决定，**必须与模型前向执行顺序一致**，否则会预取错层、掩盖失效。
- **reshard_after_forward配置**：控制FSDP2前向计算完成后是否重新分片参数。`true`（默认，ZeRO3模式）：前向完成后立即释放聚拢的参数，节省显存，但反向需要再次AllGather；`false`（ZeRO2模式）：前向完成后保留参数，反向可省一次AllGather通信，显存会相应上涨（增量因模型尺寸、所选模块与卡数而异）。在ViT模块通信量较小且显存充裕的场景下，可将ViT部分设置为`reshard_after_forward: false`以减少反向通信开销。相关参数定义位于`mindspeed_mm/fsdp/params/parallel_args.py`。

#### 其他优化

- **重计算（recompute）**：通过`recompute`参数开启、`recompute_plan.apply_modules`指定重计算的模块（使用模块路径匹配，支持精确路径、通配符和正则表达式，可精确到实例级别）。详见[FSDP2迁移指南](../features/fsdp2_developer_migration_guide.md)。
- **fully_shard切分粒度优化**：默认情况下FSDP2对每个Block单独做fully_shard，产生多次小参数通信。对于参数量较大的子模块（如MoE experts），可以通过`parallel.fsdp_plan.apply_modules`指定其单独切分，减少通信算子调用次数。详见[FSDP2迁移指南](../features/fsdp2_developer_migration_guide.md)。
- **Cast优化**：模型中可能存在不必要的精度转换（Cast）操作，主要涉及：(1) RMSNorm内部已使用fp32高精度实现，外部无需额外转fp32；(2) MoE routing（Unpermute）TorchNPU插件已修复相关bug，输入已支持output为bf16、routing_weight为fp32的组合，无需额外Cast。在Qwen3.5 MoE模型上，消融RMSNorm + Unpermute处的Cast约有5%性能收益；消融RMSNorm + Unpermute + RMSNormGated处的Cast约有10%性能收益。收益视模型结构而定，修改前应验证精度对齐。

### MCORE（Megatron）后端特性

- **异构并行**：支持不同维度的并行策略组合，详见[异构并行](../features/hetero_parallel.md)
- **序列并行**：通过切分序列维度降低单卡计算量，支持Ulysses、RingAttention、USP等算法，详见[DiT USP](../features/dit_usp.md)和[DiT Ring Attention](../features/dit_ring_attention.md)
- **融合算子与通信隐藏**：rms_norm/swiglu/flash attention等融合算子与Megatron权重更新通信隐藏，详见[迁移调优指南](./model-migration.md#性能调优)
