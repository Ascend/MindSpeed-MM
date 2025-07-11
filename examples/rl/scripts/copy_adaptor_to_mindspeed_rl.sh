#!/bin/bash

# Copy all performance files from MindSpeed MM to MindSpeed RL
# This script should be run from MindSpeed MM root directory

echo "Copying performance files from MindSpeed MM to MindSpeed RL/MindSpeed/Megatron"
echo "[Warning] Some files will be overwritten in MindSpeed RL/MindSpeed/Megatron"

while true
do
    read -r -p "Are You Sure? [Y/n] " input

    case $input in
        [yY][eE][sS]|[yY])
            echo "Copying ..."
            break
            ;;

        [nN][oO]|[nN])
            echo "Exit"
            exit 1
            ;;

        *)
            echo "Invalid input..."
            ;;
    esac
done

# Config files
cp examples/rl/code/mindspeed_rl/rl_config.py mindspeed_rl/config_cls/rl_config.py
cp examples/rl/code/mindspeed_rl/validate_config.py mindspeed_rl/config_cls/validate_config.py
cp examples/rl/code/mindspeed_rl/generate_config.py mindspeed_rl/config_cls/generate_config.py

# Dataset files
cp examples/rl/code/mindspeed_rl/mm_utils.py mindspeed_rl/datasets/mm_utils.py

# Model base files
cp examples/rl/code/mindspeed_rl/base_training_engine.py mindspeed_rl/models/base/base_training_engine.py

# Model loss files
cp examples/rl/code/mindspeed_rl/grpo_actor_loss_func.py mindspeed_rl/models/loss/grpo_actor_loss_func.py
cp examples/rl/code/mindspeed_rl/base_loss_func.py mindspeed_rl/models/loss/base_loss_func.py

# Model actor files
cp examples/rl/code/mindspeed_rl/mm_actor.py mindspeed_rl/models/mm_actor.py
cp examples/rl/code/mindspeed_rl/mm_actor_rollout_hybrid.py mindspeed_rl/models/mm_actor_rollout_hybrid.py

# Model rollout files
cp examples/rl/code/mindspeed_rl/vllm_engine.py mindspeed_rl/models/rollout/vllm_engine.py
cp examples/rl/code/mindspeed_rl/vllm_parallel_state.py mindspeed_rl/models/rollout/vllm_adapter/vllm_parallel_state.py
cp examples/rl/code/mindspeed_rl/__init__.py mindspeed_rl/models/rollout/vllm_adapter/patch/__init__.py
cp examples/rl/code/mindspeed_rl/embed_patch.py mindspeed_rl/models/rollout/vllm_adapter/patch/embed_patch.py
cp examples/rl/code/mindspeed_rl/rotary_embedding_patch.py mindspeed_rl/models/rollout/vllm_adapter/patch/rotary_embedding_patch.py

# Trainer files
cp examples/rl/code/mindspeed_rl/base.py mindspeed_rl/trainer/base.py
cp examples/rl/code/mindspeed_rl/mm_grpo_trainer_hybrid.py mindspeed_rl/trainer/mm_grpo_trainer_hybrid.py
cp examples/rl/code/mindspeed_rl/transfer_dock.py mindspeed_rl/trainer/utils/transfer_dock.py
cp examples/rl/code/mindspeed_rl/mm_transfer_dock.py mindspeed_rl/trainer/utils/mm_transfer_dock.py

# Utils files
cp examples/rl/code/mindspeed_rl/seqlen_balancing.py mindspeed_rl/utils/seqlen_balancing.py

# Worker files
cp examples/rl/code/mindspeed_rl/mm_actor_hybrid_worker.py mindspeed_rl/workers/mm_actor_hybrid_worker.py
cp examples/rl/code/mindspeed_rl/mm_integrated_worker.py mindspeed_rl/workers/mm_integrated_worker.py
cp examples/rl/code/mindspeed_rl/launcher.py mindspeed_rl/workers/scheduler/launcher.py
cp examples/rl/code/mindspeed_rl/vit_worker.py mindspeed_rl/workers/vit_worker.py
cp examples/rl/code/mindspeed_rl/megatron_sharding_manager.py mindspeed_rl/workers/resharding/megatron_sharding_manager.py

# random files
cp examples/rl/code/mindspeed_rl/megatron_random.py megatron/core/tensor_parallel/random.py
cp examples/rl/code/mindspeed_rl/mindspeed_random.py mindspeed/core/tensor_parallel/random.py
cp examples/rl/code/mindspeed_rl/dot_product_attention.py mindspeed/core/transformer/dot_product_attention.py

echo "All performance files copied successfully!"
echo "Total files copied: 27"
