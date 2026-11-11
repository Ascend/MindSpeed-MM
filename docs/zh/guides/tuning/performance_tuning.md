# 性能调优

性能优化是训练的重要环节，合理的性能调优可以显著提升模型训练效率，降低资源消耗。本文介绍MindSpeed-MM性能数据的采集、分析流程和常用调优方法。具体模型支持的特性及配置，以模型样例和对应特性文档为准。

本文从模型角度衡量性能，关注指定模型和输入数据下完成一次端到端训练step所需的时间。一个step通常对应一次优化器更新，可能包含多个梯度累积的micro-batch。多模态样本的文本长度、图像分辨率和视频帧数等会影响工作量，比较时需要保持输入和训练配置一致。

一个训练step的时间涉及以下部分：

- **数据加载**：从存储设备读取图片、视频和文本等数据，在CPU上进行预处理（如编解码），并将数据送至设备。
- **模型前向和反向**：前向计算和反向传播求导。
- **优化器**：更新模型参数。
- **模型后处理**：优化器更新后的数据处理或必要的同步，具体取决于模型。
- **通信**：单节点内设备之间以及多节点之间的数据通信。与计算重叠的通信可能被掩盖，分析时重点关注未被掩盖的部分。
- **调度下发**：Host执行指令并向设备下发算子任务的过程。

这些环节可能并行或重叠，各项累计耗时不能直接相加得到step时间；应沿时间线寻找影响step结束的等待和工作。

下图概括本文的调优流程：先采集数据，再分析瓶颈并选择调优方法；实施后验证精度、显存和性能，必要时重新采集数据分析。

```mermaid
flowchart LR
    A[采集性能数据] --> B[性能分析]
    B --> C[选择调优方法]
    C --> D[验证调优效果]
    D -->|需要进一步调优| A
```

## 采集性能数据

两个后端的采集入口不同：

- **FSDP2**：在训练YAML的`tools.profile`中配置静态采集。可参考`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`。
- **MCORE（Megatron）**：在`mindspeed_mm/tools/tools.json`中配置静态或动态采集，详见[Profiling采集工具](../../tools/tools.md#profiling采集工具)。

FSDP2静态采集示例，合并到YAML中已有的`tools`段：

```yaml
tools:
  profile:
    enable: true
    profile_type: static
    ranks: [0]
    static_param:
      level: level1
      with_cpu: true
      with_stack: false
      record_shapes: false
      with_memory: false
      save_path: ./profiling
      start_step: 10
      end_step: 11
      analyse_flag: true
```

示例采集预热后的一个step：活动步数为`end_step - start_step`，训练需运行足够长。选择稳定区间，避开初始化、checkpoint等操作。排查跨卡等待时选择相关rank；`ranks: [-1]`表示全部rank，但采集开销和数据量也会增加。需要调用栈或形状信息时再开启相应选项。

`with_memory`采集算子内存信息；`tools.memory_profile`则生成分配历史快照，使用方法见[显存调优](memory_tuning.md)。FSDP2若关闭在线解析，可运行`python mindspeed_mm/fsdp/tools/profiler.py --profiler-path ./profiling`；多机分析前应汇总各机器的采集结果。

## 性能分析流程

### 性能分析工具

采集到性能数据后，可通过[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)分析算子、通信、显存和流水线等数据，定位性能瓶颈。

多机场景下，各机器时钟可能存在偏差，影响跨卡时间线的相对位置。分析前应检查时间线对齐情况，必要时使用单卡或泳道时间校准功能，详见[MindStudio Insight系统调优指南](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/system_tuning.md)。

### 性能拆解分析方法

拿到profiling数据后，建议按照以下维度进行系统拆解：

- **计算vs通信vs空闲占比**：首先看整体时间中计算、通信和Free的比例。Free占比高通常意味着Host bound或调度问题，此时可在MindStudio Insight工具timeline中打开**下发曲线**（Flow Events）：曲线陡峭或出现明显空隙的时间段，往往就是Host来不及下发、NPU空等的位置，对照这些位置即可定位下发瓶颈并做针对性优化（参见[调度下发优化](#调度下发优化)）；通信占比高需要检查通算掩盖情况。
- **快慢卡排查**：对比各卡的耗时分布，如果存在某张卡通信或下发明显慢于其他卡，需排查是否为下发bound、负载不均衡或硬件问题。
- **算子级拆解**：将计算时间按算子类型（Cube/Vector/AI CPU）分解，识别占比最高的算子类型。Cube算子（MatMul/FA等）MFU低说明计算密度不足，Vector算子占比过高说明存在可融合空间，出现AI CPU算子则意味着该算子回退到CPU执行、通常较慢，应设法消除或替换为NPU算子。
- **子模块MFU分析**：对模型子模块（Attention、MLP等）分别计算MFU，精确定位计算瓶颈。

#### 通信耗时分析

分析通信时间在总时间中的占比，重点关注未被计算掩盖的部分，这是优化的主要目标。若通信未与计算充分重叠，可通过调整预取时机（提前发起AllGather）来改善通算掩盖。

#### 快慢卡排查

大规模集群训练时，快慢卡（部分卡明显慢于其他卡）是常见性能瓶颈。排查思路：(1)所有机器跑单机脚本对比性能数据；(2)从小集群逐步扩大，二分法定位慢节点；(3)采集集群profiling定位通信等待最长的卡。常见根因包括：下发bound（某张卡ACL接口耗时增长）、负载不均衡（多模态数据图像大小不一导致ViT部分差异大）、网络拥塞（跨Block组网或HCCL算法配置不当）、Host资源抢占（节点上非训练进程占用CPU）。

从通信timeline定位时，应先区分耗时是**真正的通信传输**还是**同步等待**：若AllGather / ReduceScatter等通信算子内存在较长的`notify_wait`，说明本卡已就绪、正在等待较慢的rank，根因多在慢卡（下发或显存重整产生异常Free）而非本卡，调整本卡的通信预取难以见效，应优先排查并消除慢卡上的下发或显存重整问题。

此外，**显存重整**也是导致长时间Free的常见原因。当显存占用过高时，分配器需要先把已有显存碎片整理、整合出一块连续空间才能完成分配，这一过程开销较大，会在timeline上表现为明显且较长时间的Free，显著拖慢训练。排查时建议在采集profile时一并开启memory采集（FSDP2下将YAML的`tools.memory_profile`段设`enable: true`），若观察到整体显存占用出现明显的上升下降（骤升骤降），通常即为触发了显存重整。由于重整对性能影响较大，应设法降低显存占用，例如降低序列长度或扩大卡数。

#### MFU分解分析

MFU（Model FLOPs Utilization，模型算力利用率）是指训练过程中实际达到的FLOPs与硬件理论峰值FLOPs之比，反映硬件算力被有效利用的程度，是衡量计算效率的常用指标。通过将MFU按模型子模块（如Attention、MLP等）和按阶段（前向、反向、重计算）进行分解，可以精确定位计算瓶颈。

分析步骤如下：

- **计算理论FLOPs**：根据模型结构和输入shape，分别计算各模块的理论FLOPs。
- **获取实际耗时**：从profiling数据中提取各模块在前向、反向各阶段的真实耗时。
- **计算子模块MFU**：`子模块MFU = 子模块FLOPs / (子模块耗时 × 理论算力)`。
- **对比定位瓶颈**：对比各子模块的MFU值，识别利用率低的模块。

实践中，不同子模块的MFU高低往往有其结构性原因：Attention层以稠密矩阵乘（如Flash Attention）为主，MFU通常较高；MoE experts层因包含permute/unpermute等非计算操作，MFU通常较低。解读分解结果时，关键是区分“结构性偏低”（如MoE routing本身就含大量非计算开销，难以改变）与“可优化的偏低”（如可融合的Vector算子、冗余的Cast、低效访存等）。后者才是优化对象。优化的目的是用合适的手段消除这些浪费、提升真实性能，而不是去抬高MFU这个数字本身。

## 性能调优方法

MindSpeed-MM提供了多种性能调优特性，可根据实际场景选择合适的策略。各特性的详细使用方法请参考[特性目录](../../features/feature_list.md)中的对应文档。部分特性与训练后端绑定，使用前请先确认当前所用后端。

### 两后端通用特性

以下为FSDP2与MCORE均支持的框架特性：

#### 显存优化

重计算以额外计算换显存；[异步激活值卸载](../../features/memory/async_activation_offload.md)将激活值卸载至Host；[ChunkLoss](../../features/optimization/chunkloss.md)分块计算loss。适用模型、配置和代价见各特性文档。

#### 融合算子

将多个连续小算子融合为单个大算子（如RotaryMul、RmsNorm、SwiGlu等），减少kernel launch开销和显存带宽消耗。融合算子的使用由框架自动应用，无需用户手动配置。

#### CPU绑核

TorchNPU提供了`CPU_AFFINITY_CONF`环境变量，用于控制CPU端任务绑核，减少上下文切换和跨NUMA延迟：`0`或未设置表示不启用；`1`表示开启粗粒度绑核；`2`表示开启细粒度绑核。开启绑核后，还可选地为指定NPU卡自定义绑核区间，例如`export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47`。详见[环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/EnvVariables/docs/zh/environment_variable_reference/CPU_AFFINITY_CONF.md)。

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

以下方法适用于FSDP2插件式后端，具体配置和适用模型以[FSDP2迁移指南](../development/fsdp2_model_migration_guide.md)及模型样例为准。

#### 显存与Host内存优化

- **重计算**：在`features`段中通过`recompute`和`recompute_plan.apply_modules`选择重计算模块，以额外计算换取显存空间。
- **Swap Core**：[Swap Core](../../features/memory/swap_core.md)统一管理张量换入换出使用的设备显存、Host锁页内存和预取策略；仅配置`swap_plan`不会启用卸载，需配合Act Stash或Op Replay等特性使用。
- **Act Stash**：[Act Stash](../../features/memory/act_stash.md)使用Swap Core卸载前向保存的激活值，反向时按需取回：设置`enable_activation_offload: true`、`activation_offload_plan.impl: stash`并指定`apply_modules`。它与原有异步激活值卸载的`legacy`实现二选一；需结合显存峰值、Host内存和拷贝开销评估。
- **Op Replay**：在重计算范围内缓存指定算子的输出，反向重计算时取回输出以跳过对应算子的重算。需开启重计算、设置`recompute_plan.use_reentrant: false`并配置`op_replay_scopes`，作用域须被重计算模块覆盖；与Act Stash同开时共享Swap Core容量。适用场景和约束见[Op Replay](../../features/optimization/op_replay.md)。
- **CCE Loss**：FSDP2的[ChunkLoss](../../features/optimization/chunkloss.md)支持CCE实现，按词表分块计算交叉熵，降低完整logits带来的显存开销。通过`features.enable_chunk_loss: true`和`chunkloss_plan.impl_type: cce`启用，可用`vocab_tile_size`调整词表块大小。当前支持`default`和`per_token_loss`，不支持带bias的输出层或动态ChunkLoss；应结合模型配置与精度验证使用。
- **ChunkMBS**：在参数聚合开销较大且显存限制batch大小时，可以按batch维度切块，在一次参数聚合后完成多个微块的计算。使用时需同时开启重计算和异步激活值卸载，ChunkMBS的`apply_modules`需被两者覆盖；配置方式和约束见[ChunkMBS](../../features/optimization/chunkmbs.md)。
- **Meta Device初始化**：设置`training.init_model_with_meta_device: true`，先在meta device上构建模型结构，减少初始化阶段的Host内存占用。具体适用情况见模型样例。

#### 通信与计算优化

- **预取（Prefetch）**：提前发起参数AllGather，尝试用计算掩盖通信。`parallel.fsdp_plan.num_to_forward_prefetch`和`num_to_backward_prefetch`分别控制前向、反向显式预取层数，默认均为`0`。自动预取根据`apply_modules`的顺序推导，配置应符合模型前向执行顺序；模型也可以自定义预取。建议结合日志和时间线检查实际顺序，并关注预取带来的显存开销。开启专家并行（EP）时，还需检查AllGather与All2All是否争用带宽。
- **前向后重新分片**：`parallel.fsdp_plan.reshard_after_forward`默认为`true`，前向后重新分片参数，节省显存，但反向需要再次聚合；设为`false`可减少反向聚合，代价是显存占用增加。该通用配置应用于计划管理的模块，调整时应同时比较通信耗时和峰值显存。
- **分片粒度**：通过`parallel.fsdp_plan.apply_modules`指定分片模块。更细的粒度可能增加通信调用次数，更粗的粒度可能增加峰值显存或影响通算掩盖；应结合模型结构和时间线选择。
- **Ulysses上下文并行**：通过`parallel.ulysses_parallel_size`切分序列计算，可降低单卡序列计算和显存压力，同时引入通信开销。具体配置与约束见[Unaligned Ulysses CP](../../features/parallel/unaligned_ulysses_cp.md)。
- **Cast优化**：检查模型中频繁出现的精度转换，结合模型实现和TorchNPU版本确认算子支持的输入、输出与累积精度，再判断是否可以消除冗余Cast。修改后应验证精度和训练结果，收益因模型而异。

### MCORE（Megatron）后端特性

- **异构并行**：组合不同维度的并行策略，详见[异构并行](../../features/parallel/hetero_parallel.md)。
- **序列并行**：根据模型和序列长度选择Ulysses、RingAttention或USP等算法，详见[DiT USP](../../features/parallel/dit_usp.md)和[DiT Ring Attention](../../features/parallel/dit_ring_attention.md)。
- **融合算子与通信隐藏**：可结合模型配置使用RMSNorm、SwiGLU、Flash Attention等融合算子及权重更新通信隐藏，详见[迁移调优指南](../development/model-migration.md#性能调优)。
