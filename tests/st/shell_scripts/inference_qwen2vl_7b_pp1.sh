#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6455
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)


LOCATION=$(pip show mindspeed 2>/dev/null | grep "^Location:" | awk '{print $2}')

echo "LOCATION: $LOCATION"
echo "BASEPATH: $BASEPATH"

mv -f "$LOCATION/mindspeed/core/transformer/dot_product_attention.py"  "$LOCATION/mindspeed/core/transformer/dot_product_attention.py_bak"

cp -rf "$BASEPATH/examples/qwen2vl/dot_product_attention.py"   "$LOCATION/mindspeed/core/transformer/dot_product_attention.py"


MM_MODEL="$BASEPATH/tests/st/run_configs/inference_qwen2vl_7B_pp1/inference_qwen2vl_7b.json"
LOAD_PATH="/home/ci_resource/models/qwen2vl_7b/qwen2vl7b_pp1"


cd $BASEPATH

TP=1
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=96
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tokenizer-type NullTokenizer \
    --vocab-size 152064 \
    --seq-length 1024 \
    --max-position-embeddings 32768 \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --seed 42 \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --use-flash-attn \
    --no-load-optim \
    --no-load-rng
"

MM_ARGS="
    --mm-model $MM_MODEL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 5000 \
"

torchrun $DISTRIBUTED_ARGS inference_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl


mv -f "$LOCATION/mindspeed/core/transformer/dot_product_attention.py_bak"  "$LOCATION/mindspeed/core/transformer/dot_product_attention.py"