#!/bin/bash

set -e
pip install tiktoken

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)
echo "BASEPATH = $BASEPATH"

fetch_and_copy_mindspeed() {
    DATE_SUFFIX=$(date +%Y%m%d)
    TARGET_PATH="/home/ci_resource/code/MindSpeed-date/MindSpeed-${DATE_SUFFIX}"

    # Check if the target mindspeed directory already exists
    if [ -d "$TARGET_PATH" ]; then
        echo "Path already exists: $TARGET_PATH"
        # Copy the mindspeed folder from the existing directory to BASEPATH
        cp -r "$TARGET_PATH/mindspeed" "$BASEPATH/"
        echo "Copied mindspeed from $TARGET_PATH to $BASEPATH"
    else
        echo "Path does not exist: $TARGET_PATH"
        # Try to clone the MindSpeed repository into the target path
        if git clone https://gitcode.com/Ascend/MindSpeed "$TARGET_PATH"; then
            echo "Clone successful: $TARGET_PATH"
            # If success, copy the mindspeed folder to BASEPATH
            cp -r "$TARGET_PATH/mindspeed" "$BASEPATH/"
            echo "Copied mindspeed from $TARGET_PATH to $BASEPATH"
        else
            echo "Clone failed, will use cache file"
            # Use a fixed cache directory (MindSpeed-26.0.0) if cloning fails
            CACHE_PATH="/home/ci_resource/code/MindSpeed-26.0.0"
            if [ -d "$CACHE_PATH/mindspeed" ]; then
                cp -r "$CACHE_PATH/mindspeed" "$BASEPATH/"
                echo "Copied mindspeed from $CACHE_PATH to $BASEPATH"
            else
                echo "Cache path does not contain mindspeed folder: $CACHE_PATH"
                exit 1
            fi
        fi
    fi
}

need_restore=false
# Check if mindspeed exists in BASEPATH
if [ -e "$BASEPATH/mindspeed" ]; then
    echo "mindspeed exists, moving to mindspeed-bak"
    # Rename the existing mindspeed to mindspeed-bak as a backup
    rm -rf "$BASEPATH/mindspeed-bak"
    mv "$BASEPATH/mindspeed" "$BASEPATH/mindspeed-bak"
    need_restore=true
fi

# Call function to fetch and copy new mindspeed
if fetch_and_copy_mindspeed; then
    echo "Successfully fetched and copied mindspeed"
else
    echo "Failed to fetch and copy mindspeed"
    # Restore backup if needed before exiting
    if [ "$need_restore" = true ]; then
        echo "Restoring backup mindspeed-bak to mindspeed"
        rm -rf "$BASEPATH/mindspeed" 2>/dev/null
        mv "$BASEPATH/mindspeed-bak" "$BASEPATH/mindspeed"
    fi
    exit 1
fi

(
  cd /home/ci_resource/models/Kimi-K2.5 && \
  cp -f \
    chat_template.jinja \
    config.json \
    configuration_deepseek.py \
    configuration_kimi_k25.py \
    generation_config.json \
    kimi_k25_processor.py \
    kimi_k25_vision_processing.py \
    preprocessor_config.json \
    tiktoken.model \
    tokenizer_config.json \
    tool_declaration_ts.py \
    ${BASEPATH}/mindspeed_mm/fsdp/models/kimik2_5/
)

CONFIG_FILE="${BASEPATH}/mindspeed_mm/fsdp/models/kimik2_5/config.json"
sed -i 's/"n_routed_experts": 384/"n_routed_experts": 64/' "$CONFIG_FILE"
sed -i 's/"num_hidden_layers": 61/"num_hidden_layers": 5/' "$CONFIG_FILE"
sed -i 's/"vt_num_hidden_layers": 27/"vt_num_hidden_layers": 5/' "$CONFIG_FILE"

# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NON_MEGATRON=true
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
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

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS $BASEPATH/mindspeed_mm/fsdp/train/trainer.py \
    $BASEPATH/tests/st/run_configs/finetune_kimik2_5/kimik2_5_config.yaml \
    2>&1 | tee logs/train_${logfile}.log

STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
GBS=`grep "global batch size" logs/train_${logfile}.log | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a logs/train_${logfile}.log
echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a logs/train_${logfile}.log

# tiktoken doesn't affect other models, keep it.
# Remove the new (temporarily fetched) mindspeed file
rm -rf "$BASEPATH/mindspeed"
echo "Remove temporarily fetched mindspeed file"
# Roll back mindspeed to the version before this script ran
if [ "$need_restore" = true ]; then
    echo "Restoring original mindspeed from backup"
    # Move the backup back to its original name
    mv "$BASEPATH/mindspeed-bak" "$BASEPATH/mindspeed"
fi
