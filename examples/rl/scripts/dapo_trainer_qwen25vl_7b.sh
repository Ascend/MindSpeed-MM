# 运行前建议释放其他资源
# pkill -9 python
# ray stop --force
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

DEFAULT_YAML="dapo_trainer_qwen25vl_7b"
YAML=${1:-$DEFAULT_YAML}
echo "Use $YAML"

ulimit -n 32768
mkdir -p logs

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export HCCL_IF_BASE_PORT=24703

# 使能 jemalloc（以OpenEuler系统为例）
# export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/libjemalloc.so.2"

NNODES=1
NPUS_PER_NODE=8
#修改为对应主节点IP(必填项)
MASTER_ADDR="xxx.xxx.xxx.xxx"
#获取当前机器IP，需要支持ip命令
CURRENT_IP=$(ip -4 addr show $(ip -o -4 route show to default | awk '{print $5}') | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          log_file="logs/dapo_train_$(date +%Y%m%d_%H%M%S).log"
          python posttrain_vlm_grpo.py --config-name $YAML 2>&1 | tee "$log_file"
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 999999