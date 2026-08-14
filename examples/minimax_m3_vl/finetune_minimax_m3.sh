ASCEND_SET_ENV=${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
source "${ASCEND_SET_ENV}"

export NON_MEGATRON=${NON_MEGATRON:-true}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1200}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}

CONFIG_PATH=${CONFIG_PATH:-examples/minimax_m3_vl/minimax_m3_config.yaml}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
LOG_DIR=${LOG_DIR:-logs}

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p "${LOG_DIR}"
torchrun ${DISTRIBUTED_ARGS} mindspeed_mm/fsdp/train/trainer.py \
    "${CONFIG_PATH}" \
    2>&1 | tee "${LOG_DIR}/train_minimax_m3_vl_${logfile}.log"
