#!/bin/bash
source /usr/local/Ascend/cann/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29505
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=4
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=4
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_DATA="./examples/mindspore/cogvideox/i2v_1.0/data.json"
MM_MODEL="./examples/mindspore/cogvideox/i2v_1.0/model_cogvideox_i2v.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="your_converted_dit_ckpt_dir"
SAVE_PATH="your_ckpt_path_to_save"

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --log_dir="msrun_log" \
    --join=True \
    --cluster_time_out=300 \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 1e-4 \
    --lr-warmup-init 1e-5 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters 5000 \
    --no-gradient-accumulation-fusion \
    --load $LOAD_PATH \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 42 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --allow-tf32 \
    --num-workers 8 \
    --seed 42 \
    --sequence-parallel \
    --qk-layernorm \
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
    --ai-framework mindspore \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME, Average Samples per Second: $SPS"