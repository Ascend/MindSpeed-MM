# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NON_MEGATRON=true
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=8
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

# cd to the repo root regardless of where the script is launched from, so that
# relative paths (trainer script / yaml / logs / plugin imports) resolve consistently
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
echo "Changing working directory to ${MINDSPEED_ROOT}"
cd "${MINDSPEED_ROOT}"

LOG_DIR=./logs/ltx2_t2av
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/mm_train_av_${TIMESTAMP}.log"

echo "Training log will be saved to: $LOG_FILE"

stdbuf -oL -eL torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/ltx2/ltx2_config_t2av.yaml 2>&1 | tee "$LOG_FILE"
