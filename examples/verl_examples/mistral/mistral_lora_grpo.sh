#!/bin/bash

source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# export ASCEND_LAUNCH_BLOCKING=1

export NPU_ASD_ENABLE=0
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_WHITELIST_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export DEVICE=npu
export WANDB_MODE=disabled

export WITHOUT_JIT_COMPILE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

export VLLM_MAX_NUM_SEQS=1
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export TRUST_REMOTE_CODE=true

export CUDA_DEVICE_MAX_CONNECTIONS=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NODE_RANK=0
export NPUS_PER_NODE=8
export NNODES=1
export USE_OPTIMIZED_MODEL=0
export TP=8

export OC_CAUSE=1

unset LOCAL_RANK

ulimit -n 32768

# 保存路径及断点续训加载路径
default_local_dir="/path/to/mistral_out"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=/path/to/gsm8k/train.parquet \
        data.val_files=/path/to/gsm8k/test.parquet \
        data.train_batch_size=8 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=False \
        data.truncation='left' \
        data.image_key=images \
        actor_rollout_ref.model.path=/path/to/Magistral-Small-2509 \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.n=2 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.fsdp_config.fsdp_size=8 \
        actor_rollout_ref.ref.strategy=fsdp2 \
        actor_rollout_ref.actor.strategy=fsdp2 \
        ++actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='["MistralDecoderLayer", "PixtralAttentionLayer"]' \
        ++actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='["MistralDecoderLayer", "PixtralAttentionLayer"]' \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.model.lora_rank=16 \
        actor_rollout_ref.model.lora_alpha=16 \
        actor_rollout_ref.model.target_modules=all-linear \
        actor_rollout_ref.rollout.load_format="auto" \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.default_local_dir=$default_local_dir \
        trainer.logger='["console"]' \
        trainer.project_name='verl_grpo_example_geo3k' \
        trainer.experiment_name='qwen2_5_vl_7b_function_rm' \
        trainer.n_gpus_per_node=$NPUS_PER_NODE \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=1 \
        trainer.total_epochs=10 \
        trainer.val_before_train=False \
        trainer.device=npu
