source /usr/local/Ascend/ascend-toolkit/set_env.sh

export NON_MEGATRON=true
export CPU_AFFINITY_CONF=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

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
logdir=logs/cosyvoice3
mkdir -p $logdir
logfile=${logdir}/finetune_cosyvoice3_$(date +%Y%m%d)_$(date +%H%M%S)
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/tasks/cosyvoice3/train.py \
    examples/fsdp2/cosyvoice3/cosyvoice3_config.yaml \
    2>&1 | tee ${logfile}.log

STEP_TIME=`grep "elapsed time per iteration" ${logfile}.log | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a ${logfile}.log