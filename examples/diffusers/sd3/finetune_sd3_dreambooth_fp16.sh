Network="StableDiffusion3Dreambooth"
model_name="stabilityai/stable-diffusion-3-medium-diffusers"
input_dir="dog"
batch_size=1
num_processors=8
max_train_steps=400
mixed_precision="fp16"
resolution=1024
gradient_accumulation_steps=1
config_file="./sd3/accelerate_config.yaml"

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}
fi

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

output_path=${cur_path}/logs

echo ${output_path}
mkdir -p ${output_path}


start_time=$(date +%s)
echo "start_time: ${start_time}"


accelerate launch --config_file ${config_file} \
  ./examples/dreambooth/train_dreambooth_sd3.py \
  --pretrained_model_name_or_path=$model_name  \
  --instance_data_dir=$input_dir \
  --output_dir=$output_path \
  --instance_prompt="a photo of sks dog" \
  --resolution=$resolution \
  --train_batch_size=$batch_size \
  --mixed_precision=$mixed_precision \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --dataloader_num_workers=0 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$max_train_steps \
  --seed="0" \
  2>&1 | tee ${output_path}/train_${mixed_precision}_sd3_dreambooth.log
wait
chmod 440 ${output_path}/train_${mixed_precision}_sd3_dreambooth.log

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
AverageIts=$(grep -oE '[0-9.]+(it/s|s/it), ' "${output_path}/train_${mixed_precision}_sd3_dreambooth.log" | \
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
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "loss=[0-9.]*" ${output_path}/train_${mixed_precision}_sd3_dreambooth.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
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
