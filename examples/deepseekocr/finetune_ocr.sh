#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

GBS=8

DATA_PATH="./data/output.jsonl"
DATA_DIR="./data"
LOAD_PATH="./ckpt/deepseek-ai/DeepSeek-OCR"
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
    --global-batch-size $GBS \
    --train-iters 1000 \
    --lr 5e-6 \
    --clip-grad 0 \
    --warmup-ratio 0 \
    --weight-decay 1e-2 \
    --data-path $DATA_PATH \
    --data-dir $DATA_DIR \
    --load $LOAD_PATH \
    --save $SAVE_PATH \
    --log_tps
"

# To ensure code security, configure trust_remote_code to default to False.
# Users need to add the following parameter and ensure the security of the models and data they download.
# --trust-remote-code \

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS examples/deepseekocr/finetune_ocr.py \
    $MODEL_ARGS \
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