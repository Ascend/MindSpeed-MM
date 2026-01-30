#!/bin/bash
source /usr/local/Ascend/cann/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=2 # 开启FSDP2时，不能置为1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
VP=1
CP=1
MBS=1
GRAD_ACC_STEP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_DATA="./examples/wan2.2/A14B/t2v/data.json"
MM_MODEL="./examples/wan2.2/A14B/t2v/pretrain_model_low.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers/transformer_2/"  # ensure the wandit weight be converted
SAVE_PATH="path to save your low noise expert wandit weight"
FSDP_CONFIG="./examples/wan2.2/A14B/fsdp2_config.yaml"

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
    --virtual-pipeline-model-parallel-size ${VP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-workers 8 \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 1e-2 \
    --lr-warmup-init 0 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters 5000 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --downcast-to-bf16 \
    --distributed-timeout-minutes 20 \
    --use-fused-rmsnorm \
    --use-torch-fsdp2 \
    --untie-embeddings-and-output-weights \
    --fsdp2-config-path ${FSDP_CONFIG} \
    --optimizer-selection fused_torch_adamw \
    --use-cpu-initialization \
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
    --ckpt-format torch_dcp \
"

logfile=wan_low_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME, Average Samples per Second: $SPS"