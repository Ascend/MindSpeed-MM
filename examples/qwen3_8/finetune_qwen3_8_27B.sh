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

# 删除triton的cache
# export TRITON_CACHE_DIR=./triton_cache
# rm -rf $TRITON_CACHE_DIR/*

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

logdir=logs/qwen38
logfile=qwen38_27B_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p $logdir
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_8/qwen3_8_27B_config.yaml \
    2>&1 | tee ${logdir}/${logfile}.log \

STEP_TIME=`grep "elapsed time per iteration" ${logdir}/${logfile}.log | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 10 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
GBS=`grep "global batch size" ${logdir}/${logfile}.log | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a ${logdir}/${logfile}.log
echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a ${logdir}/${logfile}.log
