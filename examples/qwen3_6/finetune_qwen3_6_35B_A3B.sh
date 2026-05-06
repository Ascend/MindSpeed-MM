# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200

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

logdir=logs/qwen36_35B_A3B
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p $logdir
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_6/qwen3_6_35B_A3B_config.yaml \
    2>&1 | tee $logdir/train_${logfile}.log