# 数据集和模型路径,保持为空,不需要修改
data_path=""
model_path=""

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --model_path* ]];then
        model_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path和model_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi
if [[ $model_path == "" ]];then
    echo "[Error] para \"model_path\" must be config"
    exit 1
fi

ENGINE=vllm
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=3600

# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0

# prompt&response length
max_prompt_length=1024
max_response_length=2048
max_num_batched_tokens=8192

# vllm related params
free_cache_engine=True
gpu_memory_utilization=0.5
tensor_model_parallel_size=8
enable_chunked_prefill=True
enforce_eager=False

# batch size
train_batch_size=256
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=4
use_remove_padding=True
ignore_eos=False

# training params
enable_gradient_checkpointing=True
nnodes=1
n_gpus_per_node=16
sp_size=1
parameters=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/train.parquet \
    data.val_files=$data_path/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.ignore_eos=$ignore_eos \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=$enable_chunked_prefill \
    actor_rollout_ref.rollout.enforce_eager=$enforce_eager \
    actor_rollout_ref.rollout.free_cache_engine=$free_cache_engine \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=$parameters \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen2_5_vl_32b_function_rm' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.balance_batch=False \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=30 \
    trainer.total_training_steps=150 \
    trainer.device=npu \
    trainer.val_before_train=False \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sp_size | tee train_qwen2_5_vl_32b_grpo_full.log 2>&1 &