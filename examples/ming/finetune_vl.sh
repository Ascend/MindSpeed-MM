#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DATA_PATH="./data/mllm_format_llava_instruct_data.json"
DATA_DIR="./data"
PROCESSOR_PATH="./Ming"
LOAD_PATH="./ckpt/Ming-Lite-Omni-1.5"
SAVE_PATH="save_dir"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL_ARGS="
    --num-workers 8 \
    --seed 1234 \
    --no-shuffle \
    --seq-length 2048 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 1000 \
    --lr 5e-6 \
    --clip-grad 0 \
    --warmup-ratio 0 \
    --weight-decay 1e-2 \
    --data-path $DATA_PATH \
    --data-dir $DATA_DIR \
    --processor-path $PROCESSOR_PATH \
    --load $LOAD_PATH \
    --save $SAVE_PATH \
"

# To ensure code security, configure trust_remote_code to default to False.
# Users need to add the following parameter and ensure the security of the models and data they download.
# --trust-remote-code \

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS finetune_vl.py \
    $MODEL_ARGS \
    2>&1 | tee logs/train_${logfile}.log
chmod 440 logs/train_${logfile}.log