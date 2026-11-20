#!/usr/bin/env bash
set -eo pipefail

# Run this script on every node after activating the same conda environment.
# The first IP is the master (rank 0); all nodes must use the same ordered list.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINDSPEED_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Edit this ordered list on every node; the first IP is the master (rank 0).
NODE_IPS=(
    # IP list here
    0.0.0.0
    0.0.0.1
    0.0.0.2
    0.0.0.3
)
CONFIG_PATH="${CONFIG_PATH:-${MINDSPEED_ROOT}/examples/glm5.3_flash/glm5.3_next_config_A3.yaml}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
MASTER_PORT="${MASTER_PORT:-6000}"
CANN_ENV="${CANN_ENV:-/path/to/cann/cann-9.2.0/set_env.sh}"
DRY_RUN=0

die() { echo "ERROR: $*" >&2; exit 1; }
usage() {
    echo "Usage: bash $0 [--dry-run]"
    echo "Overrides: CONFIG_PATH, NPUS_PER_NODE, MASTER_PORT, CANN_ENV, LOCAL_IP"
}
while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ "$NPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "NPUS_PER_NODE must be positive"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] && ((MASTER_PORT <= 65535)) || die "Invalid MASTER_PORT"
command -v ifconfig >/dev/null || die "The ifconfig command is required"

declare -A SEEN_IPS=()
for node_ip in "${NODE_IPS[@]}"; do
    [[ "$node_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die "Invalid IPv4 address: $node_ip"
    IFS=. read -r -a octets <<< "$node_ip"
    for octet in "${octets[@]}"; do
        [[ "$octet" == 0 || "$octet" != 0* ]] || die "IPv4 octets cannot have leading zeros: $node_ip"
        ((10#$octet <= 255)) || die "Invalid IPv4 address: $node_ip"
    done
    [[ -z "${SEEN_IPS[$node_ip]:-}" ]] || die "Duplicate IP: $node_ip"
    SEEN_IPS["$node_ip"]=1
done

NNODES=${#NODE_IPS[@]}
((NNODES > 0)) || die "IP list is empty"
MASTER_ADDR="${NODE_IPS[0]}"
LOCAL_ADDRESSES="$(ifconfig -a | awk '$1 == "inet" {addr=$2; sub(/^addr:/, "", addr); if (addr ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/) print addr}')"
mapfile -t LOCAL_IPS <<< "$LOCAL_ADDRESSES"

# LOCAL_IP can select a specific interface on a multi-homed host, but it must
# actually belong to this machine and occur in the IP list.
if [[ -n "${LOCAL_IP:-}" ]]; then
    found=0
    for address in "${LOCAL_IPS[@]}"; do
        if [[ "$address" == "$LOCAL_IP" ]]; then found=1; fi
    done
    ((found == 1)) || die "LOCAL_IP=$LOCAL_IP is not assigned to this host"
    LOCAL_IPS=("$LOCAL_IP")
fi

NODE_RANK=-1
MATCHED_IP=""
for rank in "${!NODE_IPS[@]}"; do
    for address in "${LOCAL_IPS[@]}"; do
        if [[ "${NODE_IPS[$rank]}" == "$address" ]]; then
            if ((NODE_RANK >= 0 && NODE_RANK != rank)); then
                die "Multiple list IPs belong to this host; set LOCAL_IP to select one"
            fi
            NODE_RANK=$rank
            MATCHED_IP="$address"
        fi
    done
done
((NODE_RANK >= 0)) || die "No local IP occurs in the embedded NODE_IPS list. Local IPs: ${LOCAL_IPS[*]}"
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))
[[ -f "$CONFIG_PATH" ]] || die "Config does not exist: $CONFIG_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$NPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)
TRAIN_COMMAND=(torchrun "${DISTRIBUTED_ARGS[@]}" mindspeed_mm/fsdp/train/trainer.py "$CONFIG_PATH")
echo "Node IP list: ${NODE_IPS[*]}"
echo "LOCAL_IP=$MATCHED_IP NNODES=$NNODES NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NPUS_PER_NODE=$NPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo "Working directory: $MINDSPEED_ROOT"
printf 'Launch command:'; printf ' %q' "${TRAIN_COMMAND[@]}"; printf '\n'
if ((DRY_RUN)); then exit 0; fi

[[ -r "$CANN_ENV" ]] || die "Cannot read CANN environment: $CANN_ENV"
source "$CANN_ENV"
# export HCCL_SOCKET_IFNAME=
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
command -v torchrun >/dev/null || die "torchrun not found; activate conda environment lmy_glm first"
cd "$MINDSPEED_ROOT"
mkdir -p logs
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}_$$.log"
echo "Log file: $LOG_FILE"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "$LOG_FILE"

# Keep the original performance summary, but tolerate logs without these fields.
STEP_TIME=$(awk -F 'elapsed time per iteration [(]ms[)]:' '
    NF > 1 {split($2, a, "|"); n++; if (n > 100 && n <= 200) {sum += a[1]; count++}}
    END {if (count) printf "%.1f", sum / count}' "$LOG_FILE")
GBS=$(awk -F 'global batch size:' 'NF > 1 {split($2, a, "|"); gsub(/^[ \t]+|[ \t]+$/, "", a[1]); print a[1]; exit}' "$LOG_FILE")
if [[ "$GBS" =~ ^[0-9]+([.][0-9]+)?$ && "$STEP_TIME" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    SAMPLES_PER_SECOND=$(awk -v gbs="$GBS" -v ms="$STEP_TIME" 'BEGIN {if (ms > 0) printf "%.3f", gbs * 1000 / ms}')
    if [[ -n "$SAMPLES_PER_SECOND" ]]; then
        echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a "$LOG_FILE"
        echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a "$LOG_FILE"
    fi
fi
