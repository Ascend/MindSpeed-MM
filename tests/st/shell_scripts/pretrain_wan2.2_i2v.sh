#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=2 # 开启FSDP2时，不能置为1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=29505
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

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)

MM_DATA="$BASEPATH/tests/st/run_configs/pretrain_wan2.2_i2v/data.json"
MM_MODEL="$BASEPATH/tests/st/run_configs/pretrain_wan2.2_i2v/model.json"
MM_TOOL="$BASEPATH/mindspeed_mm/tools/tools.json"
FSDP_CONFIG="$BASEPATH/tests/st/run_configs/pretrain_wan2.2_i2v/fsdp2_config.yaml"

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
    --train-iters 3 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --distributed-timeout-minutes 20 \
    --use-fused-rmsnorm \
    --use-torch-fsdp2 \
    --untie-embeddings-and-output-weights \
    --fsdp2-config-path ${FSDP_CONFIG} \
    --optimizer-selection fused_torch_adamw \
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
    --ckpt-format torch_dcp \
"

pip install diffusers==0.35.1 peft==0.17.1

torchrun $DISTRIBUTED_ARGS $BASEPATH/pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl

pip install diffusers==0.30.3 peft==0.7.1