#!/bin/bash
source /usr/local/Ascend/cann/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
config_path=examples/mcore/qwen3_5/qwen3_5_35B_config.yaml
mkdir -p logs
torchrun $DISTRIBUTED_ARGS mindspeed_mm/mcore/models/qwen3_5/pretrain_qwen3_5.py ${config_path} \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log

STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
GBS=`grep "global batch size" logs/train_${logfile}.log | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a logs/train_${logfile}.log
echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a logs/train_${logfile}.log
