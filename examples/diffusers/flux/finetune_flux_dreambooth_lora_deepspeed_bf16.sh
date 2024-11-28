Network="FluxLora"

model_name="black-forest-labs/FLUX.1-dev" 
dataset_name="pokemon-blip-captions"
batch_size=8
max_train_steps=5000
mixed_precision="bf16"
resolution=512
gradient_accumulation_steps=1
config_file="pretrain_${mixed_precision}_accelerate_config.yaml"

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
export HOST_CACHE_CAPACITY=20
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}
fi

echo ${test_path_dir}
# source ${test_path_dir}/env_npu.sh

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/logs

mkdir -p ${output_path}

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

accelerate launch --config_file ${config_file} \
  train_dreambooth_lora_flux_advanced.py \
  --pretrained_model_name_or_path=$model_name  \
  --dataset_name=$dataset_name \
  --output_dir=$output_path \
  --instance_prompt="a photo of pokemon" \
  --resolution=$resolution \
  --train_batch_size=$batch_size \
  --gradient_checkpointing \
  --mixed_precision=$mixed_precision \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$max_train_steps \
  --validation_prompt="A photo of pokemon in a bucket" \
  --validation_epochs=500 \
  --checkpointing_steps=5001 \
  --seed="0" \
  --output_dir=${output_path} > ${output_path}/train_${mixed_precision}_flux_dreambooth_lora.log 2>&1 &
wait
chmod 440 ${output_path}/train_${mixed_precision}_flux_dreambooth_lora.log
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${mixed_precision}_flux_dreambooth_lora.log | awk '{print $NF}' | sed -n '100,199p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "loss=[0-9.]*" ${output_path}/train_${mixed_precision}_flux_dreambooth_lora.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'8p'_'acc'

#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*8/'${FPS}'}')

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log