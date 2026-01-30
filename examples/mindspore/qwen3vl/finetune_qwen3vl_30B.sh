#!/bin/bash

source /usr/local/Ascend/cann/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1 # 开启FSDP2时，不能置为1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=1
export ACLNN_CACHE_LIMIT=100000
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MS_NODE_TIMEOUT=3600

# export HCCL_SOCKET_IFNAME=
# export GLOO_SOCKET_IFNAME=
# export ASCEND_RT_VISIBLE_DEVICES=

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export LOCAL_WORLD_SIZE=$NPUS_PER_NODE


MM_DATA="./examples/mindspore/qwen3vl/data_30B.json"
MM_MODEL="./examples/mindspore/qwen3vl/model_30B.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH=""
SAVE_PATH=""
LOG_PATH="msrun_log"

TP=2
PP=4
EP=2
CP=1
MBS=1
GRAD_ACC_STEP=1
SEQ_LEN=1024
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=16 #$(($MBS*$GRAD_ACC_STEP*$DP))


DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir $LOG_PATH \
    --bind_core=True \
    --join True \
"
# GPT_ARGS中模型相关参数具体配置在example/mindspore/qwen3vl/model_xb.json中，训练相关参数配置在这里
GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tokenizer-type NullTokenizer \
    --vocab-size 152064 \
    --seq-length ${SEQ_LEN} \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --lr 1.0e-6 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters 5000 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --use-flash-attn \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 8 \
    --use-distributed-optimizer \
    --bf16 \
    --sequence-parallel \
    --load $LOAD_PATH \
"


MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"


MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --variable-seq-lengths \
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 5000 \
    --save $SAVE_PATH \
    --ckpt-format torch \
"

logfile=qwen3vl_30B_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
msrun $DISTRIBUTED_ARGS pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    $MOE_ARGS \
    --ai-framework mindspore \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log


chmod 440 logs/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;