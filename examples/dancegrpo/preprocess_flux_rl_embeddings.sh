#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=19002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

LOAD_PATH="ckpt/flux"
OUTPUT_DIR="data/rl_embeddings"
PROMPT_DIR="data/prompts.txt"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GRPO_ARGS="
    --load $LOAD_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir $PROMPT_DIR \
    --sample_num 50000 \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS mindspeed_mm/tasks/rl/soragrpo/preprocess/flux_data_preprocess.py \
    $GRPO_ARGS \
    2>&1 | tee logs/preprocess_${logfile}.log
