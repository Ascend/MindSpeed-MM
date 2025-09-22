source /usr/local/Ascend/ascend-toolkit/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

TP=4
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

MM_MODEL="examples/stepvideo/t2v/inference_t2v_model.json"
LOAD_PATH="your_converted_dit_ckpt_dir"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
MM_ARGS="
 --mm-model $MM_MODEL
"

SORA_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 2e-5 \
    --min-lr 2e-5 \
    --train-iters 5010 \
    --weight-decay 0 \
    --clip-grad 1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --seed 1234 \
    --load $LOAD_PATH \
    --distributed-timeout-minutes 20 \
"

torchrun $DISTRIBUTED_ARGS inference_sora.py $MM_ARGS $SORA_ARGS