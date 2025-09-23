#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
export HCCL_BUFFSIZE=800

# 通过此配置选择使用的NPU卡,卡数需要与NPUS_PER_NODE相对应
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


MM_DATA="./examples/dancegrpo/data_dancegrpo.json"
MM_MODEL="./examples/dancegrpo/model_dancegrpo.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="ckpt/flux"
SAVE_PATH="save_dir"
HPS_REWARD_SAVE_PATH="./hps_reward.txt"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --seed 42 \
    --load $LOAD_PATH \
    --lr 1.0e-5 \
    --train-iters 300 \
    --weight-decay 0.0001 \
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
"

GRPO_ARGS="
    --cache_dir data/.cache \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --mixed_precision bf16 \
    --cfg 0.0 \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --use_hpsv2 \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --hps_reward_save $HPS_REWARD_SAVE_PATH \
    --sample_batch_size 4 \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
mkdir -p images
torchrun $DISTRIBUTED_ARGS posttrain_flux_dancegrpo.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    $GRPO_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log
chmod 440 logs/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;