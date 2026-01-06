# 本文档为MindSpeed-MM套件中对于运行脚本常用命令参数做解释说明

- [本文档为MindSpeed-MM套件中对于运行脚本常用命令参数做解释说明](#本文档为mindspeed-mm套件中对于运行脚本常用命令参数做解释说明)
  - [GPT\_ARGS下参数注释](#gpt_args下参数注释)
    - [一般参数](#一般参数)
    - [显存优化](#显存优化)
      - [重计算](#重计算)
      - [FSDP2](#fsdp2)
    - [加速特性](#加速特性)
  - [MOE\_ARGS下参数解释](#moe_args下参数解释)
  - [OUTPUT\_ARGS下参数解释说明](#output_args下参数解释说明)
  - [环境变量参数解释](#环境变量参数解释)

## GPT_ARGS下参数注释
### 一般参数
--micro-batch-size：
取值来自\${MBS},单个GPU在一次前向/反向传播中直接处理的样本数量，适应单个NPU内的内存限制。直接影响GPU显存容量。

--global-batch-size：取值来自\${GBS}，模型进行一次参数更新所使用的所有设备上的总样本数。

--num-workers: 取值为非负整数，pytorch中数据加载处理部分会启动的子进程数。设置过大会占用CPU资源，设置过小会导致模型等待数据加载过慢。

--seq_length: 序列长度，整数类型。表示模型一次能够处理的单个样本中包含的token数量。注意在启用--variable-seq-lengths时该功能失效。序列长度决定了能够捕捉的上下文信息范围，较长的序列长度可以捕捉更长的依赖关系，但会显著增加计算复杂度和内存消耗。

--normalization RMSNorm: 使用RMSNorm。推荐搭配--use-fused-rmsnorm使用。

--swiglu: store_true,使用swiglu激活函数，推荐搭配--use-fused-swiglu使用。

--lr-warmup-fraction: 0~1之间浮点数，用于学习率“预热”阶段占总步长的比例。

--weight-decay-exclude-modules: 参数级的权重衰减排除，通过配置参数名关键词（可多个）排除特定参数的权重衰减。 [详细介绍](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/parameter_lr_wd_tuning.md)

--lr-scale-modules：参数级学习率缩放，通过配置参数名关键词（可多个）来对特定参数的学习率进行缩放。 [详细介绍](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/parameter_lr_wd_tuning.md)

--clip-grad: 浮点数，默认为1，非0时启用该功能。在优化器中对权重做限制，防止loss波动过大。

--seed: 整型，随机种子。

--bf16: store_true,使用torch.bfloat16格式训练，极大降低显存消耗。

--load: str,模型权重路径，根据各example中指导填写。

--variable-seq-lengths: store_true,启用可变序列长度。

--calculate-per-sample-loss：按样本粒度计算 loss。[详细介绍](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/vlm_model_loss_calculate_type.md)

--calculate-per-token-loss：按 token 粒度计算 loss。[详细介绍](  https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/features/vlm_model_loss_calculate_type.md)

--ckpt-format torch_dcp: 保存时使用dcp格式。(详细介绍)[https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/torch_dcp.md]

--init-model-with-meta-device: 使用fsdp2的meta初始化模型，目前仅qwen3vl模型支持，详细使用请参考examples下具体模型readme.md界面。

---
### 显存优化
--tensor-model-parallel-size: 非0整数，默认为1。取值来自\${TP}。张量并行数量设置，把模型权重切分多份放到不同卡上去运算，减少单卡显存占用，但会带来额外的通信时间。

--pipeline-model-parallel-size: 非0整数，默认为1。取值来自\${PP}。流水线并行参数设置，把整个模型按阶段分到多张卡上去计算，减少单卡内存占用，但会增加通信时间，同时会引起部分卡闲时等待现象。

--context-parallel-size: 非0整数，默认为1。取值来自\${CP}。序列并行数量设置，沿着序列维度进行数据切分。主要用于长序列任务，减少单卡内存占用，会引入额外通信时间影响性能。

--context-parallel-algo: string, CP算法选择，可选范围[ulysses_cp_algo, hybrid_cp_algo, megatron_cp_algo],[详细介绍](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md)

--expert-model-parallel-size: 非0整数，默认为1。取值来自\${EP}。MOE网络中专家并行设置，把专家分配到不同卡上去进行计算。主要用来减少单张卡显存限制无法放下所有专家问题，但会引起专家负载不均衡，计算效率低的问题。

--use-distributed-optimizer: store_true,分布式优化器，将优化器状态切分到各个设备上去独立完成计算与存储。启用后可显著降低显存消耗，提升计算资源利用率。

---

#### 重计算
[详细介绍](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/pytorch/features/recompute_relative.md)

--recompute-granularity: 目前仅支持配置full用于开启全量重计算。

--recompute-method: [block, uniform]。重计算模式配置。uniform:将transformer层均匀划分组，每组大小（--recompute-num-layers）,按组存入输入和激活值。block: 前--recompute-num-layers个transformer层使用重计算，剩余层跳过。

--recompute-num-layers: 整型，使用影响如上。

---
#### FSDP2
注:启用FSDP2时，megatron各种切分策略及重计算配置均需关闭。

--fsdp2-config-path: fsdp2相关配置文件路径。

--use-cpu-initialization: 使用CPU初始化权重，需开启。

---
### 加速特性
--use-fused-swiglu: 使能相关融合算子，仅在使用swiglu时有效。

--use-fused-rmsnorm: 使能相关融合算子，仅在使用rmsnorm时有效。

--overlap-grad-reduce 与 --overlap-param-gather: 权重更新通信掩盖，仅在使能--use-distributed-optimizer时有效。[详细介绍](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md)

---
## MOE_ARGS下参数解释
--moe-token-dispatcher-type: moe网络中分发token到通信方式选择，默认为allgather,如果开启了专家并行，推荐使用alltoall。

--moe-permute-fusion: 使能permute和unpermute融合算子，加速计算。

---
## OUTPUT_ARGS下参数解释说明
--save : 取值来自SAVE_PATH,权重保存路径。
注：仅有该值配置时才会进行权重保存

--ckpt-format: 支持配置列表[torch，torch_dcp],权重保存方式。推荐优先使用torch_dcp[详细介绍](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/torch_dcp.md)。

注:1、当在使用fsdp2进行模型训练时，仅支持使用torch_dcp配置。
2、OUTPUT_ARGS下设置--ckpt-format为torch_dcp与GPT_ARGS下使能--ckpt-format torch_dcp二者作用相同，择一即可。

---
## 环境变量参数解释
所有环境变量具体解释均可在[Ascend官网](https://www.hiascend.com/)搜索查询到详细信息，以下仅展示MM套件中常用的。

| 环境初始化脚本 | 描述 | 
|-----------------------------------------|--------------------------------------------------------------------|
| `source /usr/local/Ascend/ascend-toolkit/set_env.sh`| cann安装路径，必须配置 |
| `source /usr/local/Ascend/nnal/atb/set_env.sh` | nnal安装路径 |

| 环境变量 | 描述 | 取值说明 |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| CUDA_DEVICE_MAX_CONNECTIONS | 用于控制多GPU系统下主机端并行连接的设备数量 | 需要配置为整数，取值范围`[1, 32]`；开启序列并行时需设置为1 |
| [ASCEND_SLOG_PRINT_TO_STDOUT](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0121.html) | 是否开启日志打印，开启后日志不会保存在log文件中，而是将产生的日志直接打印显示 | `0`: 关闭日志打屏<br>`1`: 开启日志打屏 |
| [ASCEND_GLOBAL_LOG_LEVEL](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0122.html) | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志 | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志 <br>注意设置为DEBUG级别后，可能会因日志流量过大影响业务性能 |
| [TASK_QUEUE_ENABLE](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_007.html) | 用于控制开启task_queue算子下发队列优化的等级 | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化 |
| [COMBINED_ENABLE](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_005.html) | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景 | `0`: 关闭<br>`1`: 开启 |
| [CPU_AFFINITY_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_033.html) | 控制CPU端算子任务的处理器亲和性，即设定任务绑核 | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核 |
| [HCCL_CONNECT_TIMEOUT](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0077.html) | 分布式场景下用于限制不同设备之间socket建链过程的超时等待时间 | 需要配置为整数，取值范围`[120, 7200]`，默认值为`120`，单位`s` |
| [PYTORCH_NPU_ALLOC_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_012.html) | 控制缓存分配器行为 | `expandable_segments`: 使能内存池扩展段功能，即虚拟内存特征  |
| [HCCL_EXEC_TIMEOUT](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0078.html) | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步 | 需要配置为整数，取值范围`[68, 17340]`，默认值为`1800`，单位`s` |
| [ACLNN_CACHE_LIMIT](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0031.html) | 配置单算子执行API在Host侧缓存的算子信息条目个数 | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000` |
| TOKENIZERS_PARALLELISM | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为 | `False`: 禁用并行分词<br>`True`: 开启并行分词 |
| [MULTI_STREAM_MEMORY_REUSE](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_016.html) | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用 |
| [NPU_ASD_ENABLE](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_029.html) | 控制是否开启Ascend Extension for PyTorch的特征值检测功能 | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| [ASCEND_LAUNCH_BLOCKING](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_006.html) | 控制算子执行时是否启动同步模式，主要用于定位代码实际出错位置，开启时会导致性能下降，仅在debug时使用 | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行 |
| NPUS_PER_NODE | 配置一个计算节点上使用的NPU数量 | 整数值（如 `1`, `8` 等）|
| [ASCEND_RT_VISIBLE_DEVICES](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0028.html) | 用于指定当前进程可见的Device，支持一次指定一个或多个Device ID | Device ID的数字组合，多个Device ID之间以英文逗号分隔 |