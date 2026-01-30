#!/bin/bash
source /usr/local/Ascend/cann/set_env.sh

export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=60010
export NPUS_PER_NODE=8

DATA_JSON="/data/datasets/t2i_dataset/data_new.jsonl"
IMAGE_FOLDER="/data/datasets/t2i_dataset/images"
LOAD_PATH="/data/weights/inclusionAI/Ming-Lite-Omni-1.5/"
PROCESSOR_PATH="."

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL_ARGS="
    --pretrained_model_name_or_path $LOAD_PATH \
    --json_path $DATA_JSON \
    --image_folder $IMAGE_FOLDER \
    --processor_path $PROCESSOR_PATH \
    --resolution 512 512 \
    --micro_batch_size 1 \
    --dataloader_num_workers 8 \
    --seed 1234 \
    --checkpointing_steps 500 \
    --max_train_steps 100000 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 50 \
    --clip_grad 0.0 \
    --lr_scheduler "constant_with_warmup" \
    --weight_decay 1e-2 \
    --weighting_scheme "logit_normal"
"

# To ensure code security, configure trust_remote_code to default to False.
# Users need to add the following parameter and ensure the security of the models and data they download.
# --trust-remote-code \


logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs


torchrun $DISTRIBUTED_ARGS finetune_t2i.py \
    $MODEL_ARGS \
    2>&1 | tee logs/train_${logfile}.log
chmod 440 logs/train_${logfile}.log