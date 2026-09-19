#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:?usage: run_training.sh CONFIG_PATH [LOG_DIR]}
LOG_DIR=${2:-logs}
NPUS_PER_NODE=${NPUS_PER_NODE:-2}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-./triton_cache}
rm -rf -- "${TRITON_CACHE_DIR:?}"/*
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${STAMP}.log"
torchrun \
  --nproc_per_node "$NPUS_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  mindspeed_mm/fsdp/train/trainer.py "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"

echo "$LOG_FILE"
