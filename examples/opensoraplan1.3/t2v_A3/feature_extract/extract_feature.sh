#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29505
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=1
ACC=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP*$ACC))

MM_DATA="examples/opensoraplan1.3/t2v_A3/feature_extract/data.json"
MM_MODEL="examples/opensoraplan1.3/t2v_A3/pretrain_t2v_model.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
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
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-15 \
    --lr-decay-style constant \
    --weight-decay 1e-2 \
    --lr-warmup-init 1e-5 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters 5000 \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 0 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --num-workers 8 \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS mindspeed_mm/tools/feature_extraction/get_sora_feature.py \
    $GPT_ARGS \
    $MM_ARGS \
    --distributed-backend nccl \
2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log