#!/bin/bash

set -e

# Record the original package version and restore it after the test case is completed
declare -A PKG_ORIGINAL_VERSIONS
for pkg in transformers triton-ascend accelerate; do
    version=$(pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    if [ -n "$version" ]; then
        PKG_ORIGINAL_VERSIONS["$pkg"]="$version"
        echo "Saved $pkg version: $version"
    else
        PKG_ORIGINAL_VERSIONS["$pkg"]=""
        echo "$pkg is not installed"
    fi
done

restore_pip_packages() {
    echo "Restoring pip packages to original versions..."
    for pkg in "${!PKG_ORIGINAL_VERSIONS[@]}"; do
        orig_version="${PKG_ORIGINAL_VERSIONS[$pkg]}"
        if [ -n "$orig_version" ]; then
            echo "Restoring $pkg to $orig_version"
            pip install "${pkg}==${orig_version}" -q
        else
            echo "Uninstalling $pkg (was not installed before)"
            pip uninstall "$pkg" -y -q
        fi
    done
}

pip install transformers==5.2.0
pip install accelerate==1.2.0

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)
echo "BASEPATH = $BASEPATH"

fetch_and_copy_mindspeed() {
    # 从 /workspace 目录下查找按分支下载的 MindSpeed 代码
    # 排除按 commit 下载的目录（commit hash 格式为 7-40 位十六进制字符，如 26ba4eb1）
    # 如果是master分支则只有MindSpeed-master一份代码；如果是${MINDSPEED_BRANCH}分支则有MindSpeed-${MINDSPEED_BRANCH}一份代码。
    # ${MINDSPEED_BRANCH}变量在ci\mm_ci_trigger.sh中定义。
    TARGET_PATH=$(ls -d /workspace/MindSpeed-* 2>/dev/null | grep -vE "MindSpeed-[0-9a-f]{7,40}$" | head -n 1)
    if [ -z "$TARGET_PATH" ] || [ ! -d "$TARGET_PATH/mindspeed" ]; then
        echo "No valid MindSpeed branch directory found in /workspace"
        return 1
    fi
    echo "Found MindSpeed branch directory: $TARGET_PATH"
    # Copy the mindspeed folder from the branch directory to BASEPATH
    cp -r "$TARGET_PATH/mindspeed" "$BASEPATH/" || {
        echo "Failed to copy mindspeed from $TARGET_PATH" >&2
        return 1
    }
    echo "Copied mindspeed from $TARGET_PATH to $BASEPATH"
    return 0
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
    $BASEPATH/tests/st/run_configs/finetune_qwen3_5_35B/qwen3_5_35B_config.yaml \
    2>&1 | tee logs/train_${logfile}.log

STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
GBS=`grep "global batch size" logs/train_${logfile}.log | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}'`
SAMPLES_PER_SECOND=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a logs/train_${logfile}.log
echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a logs/train_${logfile}.log

# Remove the new (temporarily fetched) mindspeed file
rm -rf "$BASEPATH/mindspeed"
echo "Remove temporarily fetched mindspeed file"
# Roll back mindspeed to the version before this script ran
if [ "$need_restore" = true ]; then
    echo "Restoring original mindspeed from backup"
    # Move the backup back to its original name
    mv "$BASEPATH/mindspeed-bak" "$BASEPATH/mindspeed"
fi

# Restore the pip package to its original version
restore_pip_packages
