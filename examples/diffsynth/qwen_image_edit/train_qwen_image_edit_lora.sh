#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

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

config_file="accelerate_config.yaml"
batch_size=8
max_pixels=262144
num_processors=8
mixed_precision="bf16"
learning_rate=1e-4
transformer_path="Qwen/Qwen-Image-Edit/transformer"
text_encoder_path="Qwen/Qwen-Image-Edit/text_encoder"
model_paths='[
    "'"${transformer_path}"'/diffusion_pytorch_model*.safetensors",
    "'"${text_encoder_path}"'/model*.safetensors",
    "Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"
]'
tokenizer_path="Qwen/Qwen-Image-Edit/tokenizer"
processor_path="Qwen/Qwen-Image-Edit/processor"
dataset_base_path="/path/dataset"
dataset_metadata_path="/path/dataset/metadata_edit.csv"


output_path="./logs/Qwen-Image-Edit_lora"
current_date=$(date +%Y%m%d-%H%M%S)
mkdir -p ${output_path}

start_time=$(date +%s)
echo "start_time: ${start_time}"

accelerate launch --config_file $config_file \
  ./examples/qwen_image/model_training/train.py \
  --dataset_base_path $dataset_base_path \
  --dataset_metadata_path $dataset_metadata_path \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels $max_pixels \
  --dataset_repeat 2 \
  --model_paths "$model_paths" \
  --tokenizer_path $tokenizer_path \
  --processor_path $processor_path \
  --learning_rate $learning_rate \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $output_path \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  2>&1 | tee ${output_path}/train_${mixed_precision}_ultraedit_${current_date}.log
wait
chmod 440 ${output_path}/train_${mixed_precision}_ultraedit_${current_date}.log

end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

echo "------------------ Final result ------------------"

AverageIts=$(grep -oE '[0-9.]+(it/s|s/it), ' "${output_path}/train_${mixed_precision}_ultraedit_${current_date}.log" | \
  sed -n '20,80p' | \
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

ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

echo "Final Performance images/sec : $ActualFPS"

ActualLoss=$(grep -o "loss=[0-9.]*" ${output_path}/${log_name}| awk 'END {print $NF}')

echo "Final Train Loss : ${ActualLoss}"