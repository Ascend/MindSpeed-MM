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

GPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=29505
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=4
PP=1
VP=1
CP=2
MBS=1
ACC=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP*$ACC))

MM_DATA="examples/opensoraplan1.3/t2v_A3/data_feature.json"
MM_MODEL="examples/opensoraplan1.3/t2v_A3/pretrain_t2v_model.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="your_converted_dit_ckpt_dir"
SAVE_PATH="your_ckpt_path_to_save"

DISTRIBUTED_ARGS="
    --local_worker_num $GPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir msrun_log \
    --bind_core=True
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --virtual-pipeline-model-parallel-size ${VP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
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
    --load $LOAD_PATH \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 0 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --allow-tf32 \
    --num-workers 4 \
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
    --eval-iters 10 \
    --save $SAVE_PATH \
    --ckpt-format torch \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
msrun $DISTRIBUTED_ARGS pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
PERF=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME, Average Samples per Second: $PERF"
