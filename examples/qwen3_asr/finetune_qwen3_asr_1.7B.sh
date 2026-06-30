#!/bin/bash

# 根据实际情况修改ascend-toolkit路径
source /usr/local/Ascend/cann/set_env.sh

# Runtime environment variables.
export NON_MEGATRON=true
export CUDA_DEVICE_MAX_CONNECTIONS=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HCCL_CONNECT_TIMEOUT=7200

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
log_path=logs/qwen3_asr_train_${logfile}.log
mkdir -p logs

torchrun ${DISTRIBUTED_ARGS} mindspeed_mm/fsdp/train/trainer.py \
examples/qwen3_asr/qwen3_asr_1.7B_config.yaml \
    2>&1 | tee ${log_path}

STEP_TIME=`grep "elapsed time per iteration" ${log_path} | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
GBS=`grep "global batch size" ${log_path} | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a ${log_path}
echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a ${log_path}
