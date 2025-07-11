#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ACLNN_CACHE_LIMIT=100000

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

MBS=1
GRAD_ACC_STEP=1
TP=1
PP=1
CP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)

LOCATION=$(pip show mindspeed 2>/dev/null | grep "^Location:" | awk '{print $2}')

echo "LOCATION: $LOCATION"
echo "BASEPATH: $BASEPATH"

cd $BASEPATH

MM_MODEL="$BASEPATH/tests/st/run_configs/inference_internvl2_5/inference_4B.json"


MM_ARGS="
    --mm-model ${MM_MODEL} \
"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length 4096 \
    --tokenizer-type NullTokenizer \
    --vocab-size 151674 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --bf16 \
    --use-flash-attn \
    --trust-remote-code \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 5000 \
"

torchrun $DISTRIBUTED_ARGS \
    inference_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl