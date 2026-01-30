
# 网络名称,同目录名称，需要模型审视修改
Network="StableDiffusion3Dreambooth"

scripts_path="./sd3"

# 预训练模型
model_name="stabilityai/stable-diffusion-3.5-large"
dataset_name="pokemon-blip-captions"
# input_dir="dog"
batch_size=8
num_processors=8
max_train_steps=2000
mixed_precision="fp16"
resolution=512
gradient_accumulation_steps=1
config_file="${scripts_path}/${mixed_precision}_accelerate_config.yaml"

#如果使用 input_dir="dog"，请修改dataset_name为input_dir
for para in $*; do
  if [[ $para == --model_name* ]]; then
    model_name=$(echo ${para#*=})
  elif [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --dataset_name* ]]; then
    dataset_name=$(echo ${para#*=})
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

cur_path=$(pwd)

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/logs

mkdir -p ${output_path}

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

#如果数据集为sks dog或其他，需要把 pokemon 修改为sks dog或其他
accelerate launch --config_file ${config_file} \
  ./examples/dreambooth/train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$model_name \
  --dataset_name=$dataset_name --caption_column="text" \
  --instance_prompt="A photo of pokemon" \
  --train_batch_size=$batch_size \
  --resolution=$resolution --random_flip \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --max_train_steps=$max_train_steps \
  --learning_rate=1e-05 --lr_scheduler="constant_with_warmup" --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --dataloader_num_workers=0 \
  --mixed_precision=$mixed_precision \
  --checkpointing_steps=500 \
  --seed="0" \
  --output_dir=${output_path} \
  2>&1 | tee ${output_path}/train_${mixed_precision}_sd3_dreambooth_deepspeed.log
wait
chmod 440 ${output_path}/train_${mixed_precision}_sd3_dreambooth_deepspeed.log

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
AverageIts=$(grep -oE '[0-9.]+(it/s|s/it), ' "${output_path}/train_${mixed_precision}_sd3_dreambooth_deepspeed.log" | \
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

#获取性能数据，不需要修改
# - 吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

# - 打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

# - loss值，不需要修改
ActualLoss=$(grep -o "loss=[0-9.]*" ${output_path}/train_${mixed_precision}_sd3_dreambooth_deepspeed.log | awk 'END {print $NF}')

# - 打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
# - 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'8p'_'acc'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${AverageIts}" >>${output_path}/${CaseName}.log