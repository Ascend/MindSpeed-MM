source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NON_MEGATRON=true

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
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/base_trainer.py \
    examples/fsdp2/qwen3tts/qwen3tts_config.yaml