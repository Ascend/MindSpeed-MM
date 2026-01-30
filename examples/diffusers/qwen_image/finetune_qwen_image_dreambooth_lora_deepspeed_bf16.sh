source /usr/local/Ascend/cann/set_env.sh

network="Qwen_Image"

model_name="Qwen/Qwen-Image"
dataset_name="pokemon-blip-captions"
batch_size=8
num_processors=8
max_train_steps=5000
mixed_precision="bf16"
resolution=512
gradient_accumulation_steps=1
config_file="bf16_accelerate_config.yaml"

for para in $*; do
  if [[ $para == --model_name* ]]; then
    model_name=$(echo ${para#*=})
  elif [[ $para == --dataset_name* ]]; then
    dataset_name=$(echo ${para#*=})
  elif [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --gradient_accumulation_steps* ]]; then
    gradient_accumulation_steps=$(echo ${para#*=})
  elif [[ $para == --config_file* ]]; then
    config_file=$(echo ${para#*=})
  fi
done

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1200
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CPU_AFFINITY_CONF=1
export HCCL_DETERMINISTIC=True

cur_path=$(pwd)

#创建DeviceID输出目录
output_path=${cur_path}/logs

mkdir -p ${output_path}

#训练开始时间
start_time=$(date +%s)
echo "start_time: ${start_time}"

accelerate launch --config_file $config_file \
  ./examples/dreambooth/train_dreambooth_lora_qwen_image.py \
  --pretrained_model_name_or_path=$model_name  \
  --dataset_name=$dataset_name \
  --caption_column="text" \
  --instance_prompt="a photo of pokemon" \
  --output_dir=$output_path \
  --resolution=$resolution \
  --train_batch_size=$batch_size \
  --mixed_precision=$mixed_precision \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --dataloader_num_workers=0 \
  --max_train_steps=$max_train_steps \
  --validation_epochs=2500 \
  --checkpointing_steps=5001 \
  --seed="0" \
  --gradient_checkpointing \
  2>&1 | tee ${output_path}/train_${mixed_precision}_qwen_image_dreambooth_lora.log
wait
chmod 440 ${output_path}/train_${mixed_precision}_qwen_image_dreambooth_lora.log
#训练结束时间
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印
echo "------------------ Final result ------------------"

#输出性能FPS
AverageIts=$(grep -oE '[0-9.]+(it/s|s/it), ' "${output_path}/train_${mixed_precision}_qwen_image_dreambooth_lora.log" | \
  sed -n '100,399p' | \
  awk '
  {
    match($0, /^([0-9.]+)(it\/s|s\/it)/, arr)
    num = arr[1]
    unit = arr[2]
    if (unit == "it/s") {
      value = num
    } else {
      value = 1.0 / num
    }
    sum += value
    count++
  }
  END {
      print sum / count
    }
  ')

echo "Average it/s: ${AverageIts}"
FPS=$(awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${num_processors}'*'${AverageIts}'}')

#获取性能数据
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印
echo "Final Performance images/sec : $ActualFPS"

#loss值
ActualLoss=$(grep -o "loss=[0-9.]*" ${output_path}/train_${mixed_precision}_qwen_image_dreambooth_lora.log| awk 'END {print $NF}')

#打印
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${network}_bs${BatchSize}_'8p'_'acc'

#关键信息打印到${CaseName}.log中
echo "Network = ${network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${AverageIts}" >>${output_path}/${CaseName}.log