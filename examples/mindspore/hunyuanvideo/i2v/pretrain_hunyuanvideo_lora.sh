#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

MM_DATA="./examples/mindspore/hunyuanvideo/i2v/feature_data.json"
MM_MODEL="./examples/mindspore/hunyuanvideo/i2v/model_hunyuanvideo_i2v.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./ckpt/hunyuanvideo"
SAVE_PATH="./save_ckpt/hunyuanvideo"
layerzero_config="examples/mindspore/hunyuanvideo/zero_config.yaml"

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
    --num-workers 8 \
    --lr 1e-4 \
    --min-lr 1e-4 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 0.0 \
    --lr-warmup-init 0 \
    --lr-warmup-iters 0 \
    --clip-grad 0.0 \
    --train-iters 5000 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 42 \
    --use-distributed-optimizer \
    --sequence-parallel \
"

LORA_ARGS="
    --lora-r 64 \
    --lora-alpha 64 \
    --lora-target-modules linear fc1 fc2 img_attn_qkv img_attn_proj txt_attn_qkv txt_attn_proj linear1_qkv linear1_mlp linear2 proj_out \
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
    --load $LOAD_PATH \
    --save $SAVE_PATH \
    --ckpt-format torch \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
msrun $DISTRIBUTED_ARGS pretrain_sora.py \
    $GPT_ARGS \
    $LORA_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME, Average Samples per Second: $SPS"