#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
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

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


MM_DATA="./examples/mindspore/qwen2vl/data_7b.json"
MM_MODEL="./examples/mindspore/qwen2vl/model_7b.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="ckpt/mm_path/Qwen2-VL-7B-Instruct"
SAVE_PATH="save_dir"

TP=1
PP=4
CP=1
MBS=1
GRAD_ACC_STEP=96
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir msrun_log \
    --bind_core=True
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
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
    --no-masked-softmax-fusion \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters 10000 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --use-distributed-optimizer \
    --use-flash-attn \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 8 \
    --distributed-timeout-minutes 20 \
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
    --save $SAVE_PATH \
    --ckpt-format torch \
    --log-tps \
"
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
msrun $DISTRIBUTED_ARGS pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --ai-framework mindspore \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log
chmod 440 logs/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME"
echo "Average Samples per Second: $SAMPLES_PER_SECOND"
LOG_TOKENS_PER_SECOND=`grep "tokens per sample" logs/train_${logfile}.log`
if [ "$LOG_TOKENS_PER_SECOND" ]; then
    AVERAGE_TOKENS=`grep "tokens per sample" logs/train_${logfile}.log | awk -F 'tokens per sample:' '{print$2}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
    TOKENS_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${SAMPLES_PER_SECOND}'*'${AVERAGE_TOKENS}'}'`
    echo "Consumed Tokens per Second: $TOKENS_PER_SECOND"
fi