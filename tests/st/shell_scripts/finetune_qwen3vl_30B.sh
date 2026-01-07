#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Runtime environment variables, for details please refer to the readme.
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CUDA_DEVICE_MAX_CONNECTIONS=2   # 开启FSDP2时，不能置为1
export TOKENIZERS_PARALLELISM=false
export COMBINED_ENABLE=1
export NPU_ASD_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)

MM_TOOL="$BASEPATH/mindspeed_mm/tools/tools.json"
FSDP2_PATH="$BASEPATH/examples/qwen3vl/fsdp2_config.yaml"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
config_path="$BASEPATH/tests/st/run_configs/finetune_qwen3vl_30B/qwen3vl_full_sft_30B.yaml"
sed -i \
  -e "s|^\(FSDP2_PATH: &FSDP2_PATH\) .*|\1 $FSDP2_PATH|" \
  -e "s|^\(MM_TOOL_PATH:\) .*|\1 $MM_TOOL|" \
  "$config_path"

mkdir -p logs
torchrun $DISTRIBUTED_ARGS $BASEPATH/pretrain_transformers.py ${config_path} \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log

# Print performance evaluation metrics: STEP_TIME，SAMPLES_PER_SECOND， TOKENS_PER_SECOND
chmod 440 logs/train_${logfile}.log
SAVE_PATH=$(grep "saving checkpoint at iteration" logs/train_${logfile}.log | tail -n 1 | awk '{for(i=1;i<=NF;i++){if($i=="to"){print $(i+1);break}}}')
[ -d "$SAVE_PATH" ] && (find "$SAVE_PATH" -type d -exec chmod 750 {} \; && find "$SAVE_PATH" -type f -exec chmod 640 {} \; && echo "Success: Modified permissions for $SAVE_PATH") || echo "Warning: Invalid save path: $SAVE_PATH"
STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
GBS=`grep "consumed samples:" logs/train_${logfile}.log | tail -n 1 | awk -F '|' '{split($1, a, "iteration"); split(a[2], b, "/"); iter=b[1]+0; split($2, c, ":"); samp=c[2]+0; if(iter!=0) printf("%.2f", samp/iter); else print "N/A"}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME"
echo "Average Samples per Second: $SAMPLES_PER_SECOND"
LOG_TOKENS_PER_SECOND=`grep "tokens per sample" logs/train_${logfile}.log`
if [ "$LOG_TOKENS_PER_SECOND" ]; then
    AVERAGE_TOKENS=`grep "tokens per sample" logs/train_${logfile}.log | awk -F 'tokens per sample:' '{print$2}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
    TOKENS_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${SAMPLES_PER_SECOND}'*'${AVERAGE_TOKENS}'}'`
    echo "Consumed Tokens per Second: $TOKENS_PER_SECOND"
fi