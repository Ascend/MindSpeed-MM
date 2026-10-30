# Model Configuration

## Applicable Backends

This document applies to the Mcore and Mcore-FSDP2 backends.

## Configuration Description

Model configuration can be performed in two ways.

Method 1: Configure model-related parameters in a yaml file. Take [examples/qwen3vl/qwen3vl_full_sft_30B.yaml](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/qwen3vl/qwen3vl_full_sft_30B.yaml) as an example:

```YAML
### Model-related configuration
model:
  model_id: qwen3_vl_moe
  init_from_hf_path: *HF_MODEL_LOAD_PATH
  image_encoder:
    vision_encoder:
      model_id: qwen3vit
      num_layers: 27  # This parameter is used by Megatron for validation and does not take effect at runtime. To modify it, edit config.json in the HF_MODEL_LOAD_PATH directory.
      hidden_size: 1152  # This parameter is used by Megatron for validation and does not take effect at runtime.
      num_attention_heads: 16  # This parameter is used by Megatron for validation and does not take effect at runtime.
      freeze: true
      attn_implementation: flash_attention_2
      attn_layout: TND
      synchronize_per_layer: true
    vision_projector:
      model_id: lnmlp
      num_layers: 1
      freeze: true
  text_decoder:
    model_id: qwen3lm
    num_layers: 48  # This parameter is used by Megatron for validation and does not take effect at runtime. To modify it, edit config.json in the HF_MODEL_LOAD_PATH directory.
    hidden_size: 2048  # This parameter is used by Megatron for validation and does not take effect at runtime.
    num_attention_heads: 32  # This parameter is used by Megatron for validation and does not take effect at runtime.
    max_position_embeddings: 262144  # This parameter is used by Megatron for validation and does not take effect at runtime.
    freeze: false
    use_npu_fused_moe: false
    attn_implementation: flash_attention_2
    attn_layout: TND
    is_causal: false
    activation_offload: false
    synchronize_per_layer: true
  loss_cfg:
      compute_mode: default
      chunk_size: 1024
      router_aux_loss_coef: 0.0
      loss_type: default
  patch:
      clip_grad_async: true
      bridge_patch: true
```

Method 2: Configure in a separate json file. Take [examples/wan2.2/A14B/t2v/pretrain_model_high.json](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/examples/wan2.2/A14B/t2v/pretrain_model_high.json) as an example.

```json
{
    "load_video_features": false,                           // Whether to load video features already encoded by the VAE directly
    "load_text_features": false,                            // Whether to load text_embeds already encoded by the text_encoder directly
    "task": "t2v",
    "diffusion": {                                          // Diffusion model configuration
        "model_id": "wan_flow_match_scheduler",             // model_id is used for model class mapping, mindspeed_mm/models/diffusion/diffusion_model/DIFFUSION_MODEL_MAPPINGS
        "num_train_timesteps": 1000,                        // The following parameters are the actual initialization parameters of DiffusionModel
        "shift": 5,
        "sigma_min": 0.0,
        "extra_one_step": true,
        "max_timestep_boundary": 0.417,
        "min_timestep_boundary": 0.0
    },
    "predictor": {                                           // DiT model configuration
        "model_id": "wandit",                                // model_id is used for model class mapping, mindspeed_mm/models/predictor/predict_model/PREDICTOR_MODEL_MAPPINGS
        "dtype": "bf16",                                     // The following parameters are the actual initialization parameters of PredictModel
        "model_type": "t2v",
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 16,
        "hidden_size": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "img_dim": 1280,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "pipeline_num_layers": [10,10,10,10],
        "qk_norm": true,
        "qk_norm_type": "rmsnorm",
        "cross_attn_norm": true,
        "eps": 1e-6,
        "max_seq_len": 1024,
        "attention_async_offload": false,
        "use_fused_rmsnorm":true
    },
    "ae": {                                                   // AE model configuration
        "model_id": "wan_video_vae",                          // model_id is used for model class mapping, mindspeed_mm/ae/base/AE_MODEL_MAPPINGS
        "from_pretrained": "Wan2.2-T2V-A14B-Diffusers/vae",   // Path for loading AE model weights
        "dtype": "bf16",                                      // The following parameters are the actual initialization parameters of AEModel
        "enable_tiling": false,
        "tiling_param": {
            "tile_sample_min_height": 256,
            "tile_sample_min_width": 256,
            "tile_sample_stride_height": 192,
            "tile_sample_stride_width": 192
        },
        "norm_latents": true,
        "norm_mode": "channel_specified_shift_scale",
        "do_sample": false
    },
    "text_encoder": {                                                // TextEncoder model configuration
        "model_id": "UMT5",                                          // model_id must be set to the actual transformers model class
        "dtype": "bf16",             
        "hub_backend": "hf",                                         // hf backend
        "from_pretrained": "Wan2.2-T2V-A14B-Diffusers/text_encoder"  // Path for loading TextEncoder model weights and configuration
    }
    "patch": {                                                       // patch
        "clip_grad_async": true,
        "get_dist_model_load_from_pt": true
    },
    "encoder_offload_interval": 8,                                   // encoder_offload_interval optimization feature configuration
}
```

The parameters configured in the json file above are the same as those configured in the yaml file; only the format differs. In the launch script, Method 1 requires the `config.yaml` file, and Method 2 requires the `model.json` file.
Note that understanding models support both methods above, while generation models currently support only Method 2.

Method 1:

```Shell
config_path=examples/qwen3vl/qwen3vl_full_sft_30B.yaml
torchrun $DISTRIBUTED_ARGS pretrain_transformers.py ${config_path} \
    --distributed-backend nccl
```

Method 2:

```Shell
......
MM_MODEL="./examples/wan2.2/A14B/t2v/pretrain_model_high.json"
MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"
......
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log
```

Note:
For classes initialized with `TransformersModel(mindspeed_mm/models/TransformersModel)`, the model configuration comes from `config.json` in the Hugging Face weight path. In this case, `model.json` is not used as the configuration for actual model initialization, but only for validating model parameters in the Megatron backend.

Taking the Qwen3VL model as an example, its actual configuration comes from `Qwen3-VL-*B-Instruct/config.json` in the Hugging Face weight path,

However, some configurations are overwritten by the `overwrite_transformer_config`(mindspeed_mm/models/transformers/qwen3vl/Qwen3VLFSDP2Mixin) function, mainly including parameters such as `attn_implementation`,`attn_layout`, `is_causal`, and `activation_offloadsynchronize`.
