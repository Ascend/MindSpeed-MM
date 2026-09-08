source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NON_MEGATRON=true
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-torch}"
export HCCL_DETERMINISTIC=True
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"

NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
config_path="${CONFIG_PATH:-${SCRIPT_DIR}/minimax_h3_fl2va.yaml}"

cd "${MINDSPEED_ROOT}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/train_${logfile}.log"
mkdir -p "${LOG_DIR}"
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    ${config_path} \
    2>&1 | tee "${LOG_FILE}"

STEP_TIME=$(grep "elapsed time per iteration" "${LOG_FILE}" | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}')
GBS=$(grep "global batch size" "${LOG_FILE}" | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}')
SAMPLES_PER_SECOND=$(awk -v gbs="${GBS}" -v step_time="${STEP_TIME}" 'BEGIN{if (step_time != "") printf "%.3f\n", gbs*1000/step_time}')
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a "${LOG_FILE}"
echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a "${LOG_FILE}"
