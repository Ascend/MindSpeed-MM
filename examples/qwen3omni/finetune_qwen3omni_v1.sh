#!/bin/bash

source /usr/local/Ascend/cann/set_env.sh
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export ASCEND_LAUNCH_BLOCKING=0
export HCCL_CONNECT_TIMEOUT=1200
export TASK_QUEUE_ENABLE=2
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 当前脚本双机拉起配置仅作参考，请根据实际情况修改
export GLOO_SOCKET_IFNAME="Your SOCKET IFNAME"
# 当前脚本双机8卡，请根据实际情况指定设备
NPUS_PER_NODE=8
MASTER_ADDR=<master_ip_address>
MASTER_PORT=6000
NNODES=2
NODE_RANK=0 # 主节点为0，从节点为1，除了该参数，其他配置都要保持一致

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3omni/qwen3omni_config_v1.yaml \
    2>&1 | tee logs/train_${logfile}.log