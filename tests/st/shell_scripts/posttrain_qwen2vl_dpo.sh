#!/bin/bash
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=2
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)
LOCATION=$(pip show mindspeed 2>/dev/null | grep "^Location:" | awk '{print $2}')
echo "LOCATION: $LOCATION"
echo "BASEPATH: $BASEPATH"

MM_DATA="$BASEPATH/tests/st/run_configs/posttrain_qwen2vl_dpo/data_72b_dpo.json"
MM_MODEL="$BASEPATH/tests/st/run_configs/posttrain_qwen2vl_dpo/model_72b.json"
MM_TOOL="$BASEPATH/mindspeed_mm/tools/tools.json"

TP=2
# 注意修改MM_MODEL里面PP配置，详见readme
PP=4
CP=1
MBS=1
GRAD_ACC_STEP=2
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# GPT_ARGS中模型相关参数具体配置在example/qwen2vl/model_72b.json中，训练相关参数配置在这里
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
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --lr 5.0e-6 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters 3 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --bf16 \
    --variable-seq-lengths \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --use-flash-attn \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 8 \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 5000 \
"

DPO_ARGS="
    --dpo-beta 0.1 \
    --dpo-loss-type sigmoid \
    --dpo-label-smoothing 0.0 \
    --pref-ftx 0.0 \
"

torchrun $DISTRIBUTED_ARGS \
    $BASEPATH/posttrain_qwen2vl_dpo.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    $DPO_ARGS \
    --distributed-backend nccl \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2