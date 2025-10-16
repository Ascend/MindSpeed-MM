# 本文档为MindSpeed-MM套件中对于运行脚本常用命令参数做解释说明

- [本文档为MindSpeed-MM套件中对于运行脚本常用命令参数做解释说明](#本文档为mindspeed-mm套件中对于运行脚本常用命令参数做解释说明)
  - [GPT\_ARGS下参数注释](#gpt_args下参数注释)
    - [一般参数](#一般参数)
    - [显存优化](#显存优化)
      - [重计算](#重计算)
      - [FSDP2](#fsdp2)
    - [加速特性](#加速特性)
  - [MOE\_ARGS下参数解释](#moe_args下参数解释)
  - [环境变量参数解释](#环境变量参数解释)

## GPT_ARGS下参数注释
### 一般参数
--micro-batch-size：
取值来自\${MBS},单个GPU在一次前向/反向传播中直接处理的样本数量，适应单个NPU内的内存限制。直接影响GPU显存容量。

--global-batch-size：取值来自\${GBS}，模型进行一次参数更新所使用的所有设备上的总样本数

--num-workers: 取值为非负整数，pytorch中数据加载处理部分会启动的子进程数。设置过大会占用CPU资源，设置过小会导致模型等待数据加载过慢。

--seq_length: 序列长度，整数类型。表示模型一次能够处理的单个样本中包含的token数量。注意在启用--variable-seq-lengths时该功能失效。序列长度决定了能够捕捉的上下文信息范围，较长的序列长度可以捕捉更长的依赖关系，但会显著增加计算复杂度和内存消耗。

--normalization RMSNorm: 使用RMSNorm。推荐搭配--use-fused-rmsnorm使用。

--swiglu: store_true,使用swiglu激活函数，推荐搭配--use-fused-swiglu使用。

--lr-warmup-fraction: 0~1之间浮点数，用于学习率“预热”阶段占总步长的比例。

--clip-grad: 浮点数，默认为1，非0时启用该功能。在优化器中对权重做限制，防止loss波动过大。

--seed: 整形，随机种子。

--bf16: store_true,使用torch.bfloat16格式训练，极大降低显存消耗。

--load: str,模型权重路径，根据各example中指导填写

--variable-seq-lengths: store_true,启用可变序列长度

---
### 显存优化
--tensor-model-parallel-size: 非0整数，默认为1。取值来自\${TP}。张量并行数量设置，把模型权重切分多份放到不同卡上去运算，减少单卡显存占用，但会带来额外的通信时间。

--pipeline-model-parallel-size: 非0整数，默认为1。取值来自\${PP}。流水线并行参数设置，把整个模型按阶段分到多张卡上去计算，减少单卡内存占用，但会增加通信时间，同时会引起部分卡闲时等待现象。

--context_parallel-size: 非0整数，默认为1。取值来自\${CP}。序列并行数量设置，沿着序列维度进行数据切分。主要用于长序列任务，减少单卡内存占用，会引入额外通信时间影响性能。

--context-parallel-algo: string, CP算法选择，可选范围[ulysses_cp_algo, hybrid_cp_algo, megatron_cp_algo],[详细介绍](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md)

--expert-model-parallel-size: 非0整数，默认为1。取值来自\${EP}。MOE网络中专家并行设置，把专家分配到不同卡上去进行计算。主要用来减少单张卡显存限制无法放下所有专家问题，但会引起专家负载不均衡，计算效率低的问题。

--use-distributed-optimizer: store_true,分布式优化器，将优化器状态切分到各个设备上去独立完成计算与存储。启用后可显著降低显存消耗，提升计算资源利用率。

---

#### 重计算
[详细介绍](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/recompute_relative.md)

--recompute-granularity: 目前仅支持配置full。开启重计算

--recompute-method: [block, uniform]。重计算模式配置。uniform:将transformer层均匀划分组，每组大小（--recompute-num-layers）,按组存入输入和激活值。block: 前--recompute-num-layers个transformer层使用重计算，剩余层跳过。

--recompute-num-layers: 整形，使用影响如上。

---
#### FSDP2
注:启用FSDP2时，megatron各种切分策略及重计算配置均需关闭

--fsdp2-config-path: fsdp2相关配置文件路径

--use-cpu-initialization: 使用CPU初始化权重，需开启。

---
### 加速特性
--use-fused-swiglu: 使能相关融合算子，仅在使用swiglu时有效。

--use-fused-rmsnorm: 使能相关融合算子，仅在使用rmsnorm时有效。

--overlap-grad-reduce 与 --overlap-param-gather: 权重更新通信掩盖，仅在使能--use-distributed-optimizer时有效。[详细介绍](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/async-ddp-param-gather.md)

---
## MOE_ARGS下参数解释
--moe-token-dispatcher-type: moe网络中分发token到通信方式选择，默认为allgather,如果开启了专家并行，推荐使用alltoall

--moe-permute-fusion: 使能permute和unpermute融合算子，加速计算。

---
## 环境变量参数解释
所有环境变量具体解释均可在[Ascend官网](https://www.hiascend.com/)搜索查询到详细信息，以下仅展示MM套件中常用的。

source /usr/local/Ascend/ascend-toolkit/set_env.sh: cann安装路径，必须配置。
export CUDA_DEVICE_MAX_CONNECTIONS: 整形，[1, 32]可选。在多GPU系统下，可以使用该变量来控制主机端并行连接多设备数量。注意当开启了序列并行时，需要置为1。

ASCEND_SLOG_PRING_TO_STDOUT: 0 or 1,是否开启日志打印。开启后日志不会保存在log文件中，而是将产生的日志直接打印显示，默认置为0。

ASCEND_GLOBAL_LOG_LEVEL: 整形，[0, 4]可选。设置日志级别，仅支持调试日志。
* 0: DEBUG
* 1: INFO
* 2: WARNING
* 3: ERROR,默认
* 4: NULL
注意设置为DEBUG级别后，可能会因为日志流量过大影响业务性能

TASK_QUEUE_ENABLE: 整形，[0, 2]可选，配置task_queue算子下发队列是否开启及优化等级，推荐配置为2。[详细介绍](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_007.html)

COMBINED_ENABLE: 0 or 1,设置combined标志，用于优化非连续两个算子组合类场景，推荐配置为1。

CPU_AFFINITY_CONF: 粗\细粒度绑核功能设置，推荐配置为1。[详细介绍](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_033.html)

HCCL_CONNECT_TIMEOUT: 整形，[120, 7200]可选，分布式场景下用于限制不同设备间socket建链过程的超时等待时间，单位为秒。[详细介绍](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0077.html)

NPU_ASD_ENABLE: 0 or 1,是否开启Ascend EXTENSION FOR PyTorch的特征值检测功能，推荐置为0。[详细介绍](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_029.html)

ASCEND_LAUNCH_BLOCKING: 0 or 1,置为1时使能。控制算子执行时是否启动同步模式，主要用于定位代码实际出错位置，开启时会导致性能下降，仅在debug时使用。

ACLNN_CACHE_LIMIT: 整形，[1, 10000000], 默认值为10000，用于配置单算子执行API在Host侧缓存的算子信息条目个数。

PYTORCH_NPU_ALLOC_CONF=“expandable_segments:True”: [详细介绍](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_012.html)