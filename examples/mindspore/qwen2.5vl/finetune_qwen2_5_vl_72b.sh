#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# The current configuration is for reference only. Please modify it according to your actual needs.
NPUS_PER_NODE=8
# Note: For multi-machine situation, please create file hostfile.txt based on the actual IP with one IP per line.
HOSTFILE="examples/mindspore/qwen2.5vl/hostfile.txt"
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')  # The first line is master address.
MASTER_PORT=6000
NODE_ADDR=`hostname -I | awk '{for(i=1;i<=NF;i++)print $i}' | grep ${MASTER_ADDR%.*}. | awk -F " " '{print$1}'`
NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
NNODES=$(cat $HOSTFILE | wc -l)
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export LOCAL_WORLD_SIZE=$NPUS_PER_NODE
echo $MASTER_ADDR
echo $NODE_ADDR
echo $NODE_RANK
echo $NNODES


MM_DATA="./examples/mindspore/qwen2.5vl/data_72b.json"
MM_MODEL="./examples/mindspore/qwen2.5vl/model_72b.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-72B-Instruct"
SAVE_PATH="save_dir"
LOG_PATH="msrun_log"

TP=2
PP=8
CP=1
MBS=1
GRAD_ACC_STEP=96
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir $LOG_PATH \
    --bind_core=True \
    --join True \
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tokenizer-type NullTokenizer \
    --vocab-size 152064 \
    --seq-length 1024 \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters 10000 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --use-distributed-optimizer \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 8 \
    --use-flash-attn \
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
"
logfile=train_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
msrun $DISTRIBUTED_ARGS pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    2>&1 | tee logs/${logfile}.log
chmod 440 logs/${logfile}.log
chmod 440 ${LOG_PATH}/*.log
chmod -R 640 $SAVE_PATH
STEP_TIME=`grep "elapsed time per iteration" logs/${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME"
echo "Average Samples per Second: $SAMPLES_PER_SECOND"
LOG_TOKENS_PER_SECOND=`grep "tokens per sample" logs/${logfile}.log`
if [ "$LOG_TOKENS_PER_SECOND" ]; then
    AVERAGE_TOKENS=`grep "tokens per sample" logs/${logfile}.log | awk -F 'tokens per sample:' '{print$2}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
    TOKENS_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${SAMPLES_PER_SECOND}'*'${AVERAGE_TOKENS}'}'`
    echo "Consumed Tokens per Second: $TOKENS_PER_SECOND"
fi