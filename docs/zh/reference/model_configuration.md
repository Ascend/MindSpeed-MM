# 模型配置

## 适用后端

该文档适用于Mcore以及Mcore-FSDP2后端。

## 配置说明

模型配置有两种方式可以进行配置，

方式一：在yaml文件中进行model相关参数配置，以[examples/qwen3vl/qwen3vl_full_sft_30B.yaml](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/qwen3vl/qwen3vl_full_sft_30B.yaml)为例：

```YAML
### 模型相关配置
model:
  model_id: qwen3_vl_moe
  init_from_hf_path: *HF_MODEL_LOAD_PATH
  image_encoder:
    vision_encoder:
      model_id: qwen3vit
      num_layers: 27  # 此参数为megatron校验参数，运行中不生效，如需修改请修改HF_MODEL_LOAD_PATH目录下的config.json
      hidden_size: 1152  # 此参数为megatron校验参数，运行中不生效
      num_attention_heads: 16  # 此参数为megatron校验参数，运行中不生效
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
    num_layers: 48  # 此参数为megatron校验参数，运行中不生效，如需修改请修改HF_MODEL_LOAD_PATH目录下的config.json
    hidden_size: 2048  # 此参数为megatron校验参数，运行中不生效
    num_attention_heads: 32  # 此参数为megatron校验参数，运行中不生效
    max_position_embeddings: 262144  # 此参数为megatron校验参数，运行中不生效
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

方式二：在单独的json文件中进行配置，以[examples/wan2.2/A14B/t2v/pretrain_model_high.json](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.0.0/examples/wan2.2/A14B/t2v/pretrain_model_high.json)为例

```json
{
    "load_video_features": false,                           // 是否直接加载 vae 编码后的 video 特征
    "load_text_features": false,                            // 是否直接加载 text_encoder 编码后的 text_embeds
    "task": "t2v",
    "diffusion": {                                          // diffusion 模型配置
        "model_id": "wan_flow_match_scheduler",             // model_id 用于模型类映射，mindspeed_mm/models/diffusion/diffusion_model/DIFFUSION_MODEL_MAPPINGS
        "num_train_timesteps": 1000,                        // 下面几行参数为实际的 DiffusionModel 的初始化参数
        "shift": 5,
        "sigma_min": 0.0,
        "extra_one_step": true,
        "max_timestep_boundary": 0.417,
        "min_timestep_boundary": 0.0
    },
    "predictor": {                                           // DiT模型配置
        "model_id": "wandit",                                // model_id 用于模型类映射，mindspeed_mm/models/predictor/predict_model/PREDICTOR_MODEL_MAPPINGS
        "dtype": "bf16",                                     // 下面几行参数为实际的 PredictModel 的初始化参数
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
    "ae": {                                                   // AE 模型配置
        "model_id": "wan_video_vae",                          // model_id 用于模型类映射，mindspeed_mm/ae/base/AE_MODEL_MAPPINGS
        "from_pretrained": "Wan2.2-T2V-A14B-Diffusers/vae",   // AE 模型权重加载路径
        "dtype": "bf16",                                      // 下面几行参数为实际的 AEModel 的初始化参数
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
    "text_encoder": {                                                // TextEncoder 模型配置
        "model_id": "UMT5",                                          // model_id 需要配置成实际的transformers模型类
        "dtype": "bf16",             
        "hub_backend": "hf",                                         // hf 后端
        "from_pretrained": "Wan2.2-T2V-A14B-Diffusers/text_encoder"  // TextEncoder模型权重和配置加载路径
    }
    "patch": {                                                       // patch
        "clip_grad_async": true,
        "get_dist_model_load_from_pt": true
    },
    "encoder_offload_interval": 8,                                   // encoder_offload_interval 优化特性配置 
}
```

上述json文件中配置的参数与yaml文件中配置的参数是一致的，只是格式不同。在拉起脚本中，方式一需要指定config.yaml文件，方式二需要指定model.json文件。
注意:理解类型模型支持上述两种方式，生成模型暂时支持方式二。
方式一:

```Shell
config_path=examples/qwen3vl/qwen3vl_full_sft_30B.yaml
torchrun $DISTRIBUTED_ARGS pretrain_transformers.py ${config_path} \
    --distributed-backend nccl
```

方式二:

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

注意：
用 `TransformersModel(mindspeed_mm/models/TransformersModel)`初始化的类，模型配置来自于 huggingface权重路径下的config.json，此时model.json不会作为实际模型初始化的配置，只是为了Megatron底层对模型参数的校验。

以 Qwen3VL 模型为例，其实际配置来自于 huggingface 权重路径Qwen3-VL-\*B-Instruct/config.json，

但是部分配置会通过overwrite_transformer_config(mindspeed_mm/models/transformers/qwen3vl/Qwen3VLFSDP2Mixin)函数进行复写，主要为 `attn_implementation` `attn_layout` `is_causal`activation_offloadsynchronize等参数。
