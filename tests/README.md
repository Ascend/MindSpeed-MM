# MindSpeed MM 测试用例编写指南

本文档详细说明如何为MindSpeed MM贡献DT用例。

## 一、背景与参考

### 1.1 MindSpeed MM仓CI门禁代码相关路径

| 用途 | 路径 |
| ------ | ------ |
| 测试用例 | `MindSpeed-MM/tests` |
| CI启动代码 | `MindSpeed-MM/ci` |

### 1.2 CI门禁范围

CI门禁看护以下两项指标：

1. **功能**：代码能够正常运行
2. **性能**：性能劣化不得超过5%

---

## 二、CI门禁看护列表

PR合入前都须通过全量CI门禁用例测试。

### 2.1 ST（系统测试）看护列表

> **说明**：ST用例看护性能指标，性能劣化不得超过5%。

| Module | Features | Scripts |
| :------- | :--------- | :-------- |
| **Pretrain** | CogVideoX T2V, TP=2, CP=2, Ulysses CP | [pretrain_cogvideox_t2v_1_0.sh](st/shell_scripts/pretrain_cogvideox_t2v_1_0.sh) |
| | CogVideoX I2V, TP=2, PP=2, CP=2, Ulysses CP | [pretrain_cogvideox_i2v_1.5.sh](st/shell_scripts/pretrain_cogvideox_i2v_1.5.sh) |
| | HunyuanVideo T2V, TP=4, CP=1 | [pretrain_hunyuanvideo_t2v.sh](st/shell_scripts/pretrain_hunyuanvideo_t2v.sh) |
| | OpenSoraPlan 1.3, TP=2, CP=2 | [pretrain_opensoraplan1_3.sh](st/shell_scripts/pretrain_opensoraplan1_3.sh) |
| | Wan2.1 T2V, FSDP2 | [pretrain_wan2.1_t2v.sh](st/shell_scripts/pretrain_wan2.1_t2v.sh) |
| | Wan2.2 I2V, FSDP2 | [pretrain_wan2.2_i2v.sh](st/shell_scripts/pretrain_wan2.2_i2v.sh) |
| **Finetune** | Qwen2VL 7B, TP=1, PP=4 | [finetune_qwen2vl_7B.sh](st/shell_scripts/finetune_qwen2vl_7B.sh) |
| | Qwen2.5VL 7B, TP=2, PP=2 | [finetune_qwen2_5_vl_7b.sh](st/shell_scripts/finetune_qwen2_5_vl_7b.sh) |
| | DeepSeekVL2, TP=2, PP=2 | [finetune_deepseekvl2.sh](st/shell_scripts/finetune_deepseekvl2.sh) |
| | Qwen3Omni, FSDP2 | [finetune_qwen3omni.sh](st/shell_scripts/finetune_qwen3omni.sh) |
| | Qwen3VL 30B, FSDP2 | [finetune_qwen3vl_30B.sh](st/shell_scripts/finetune_qwen3vl_30B.sh) |
| **Posttrain** | Qwen2VL, DPO, TP=2, PP=4 | [posttrain_qwen2vl_dpo.sh](st/shell_scripts/posttrain_qwen2vl_dpo.sh) |
| **Inference** | Qwen2VL 7B, PP=1 | [inference_qwen2vl_7b_pp1.sh](st/shell_scripts/inference_qwen2vl_7b_pp1.sh) |
| | Qwen2VL 7B, PP=4 | [inference_qwen2vl_7b_pp4.sh](st/shell_scripts/inference_qwen2vl_7b_pp4.sh) |
| | CogVideoX T2V 1.5 | [inference_cogvideox_t2v_1.5.sh](st/shell_scripts/inference_cogvideox_t2v_1.5.sh) |
| | InternVL2.5 | [inference_internvl2_5.sh](st/shell_scripts/inference_internvl2_5.sh) |
| | Wan2.2 T2V, CP=2 | [inference_wan2.2_t2v.sh](st/shell_scripts/inference_wan2.2_t2v.sh) |

### 2.2 UT（单元测试）看护列表

> **说明**：UT用例看护功能指标，确保代码能够正常运行。

| Module | Features | Scripts |
| :------- | :--------- | :-------- |
| **Loss** | Chunk Loss | [test_chunkloss.py](ut/loss/test_chunkloss.py) |
| **Tools** | Profiler性能分析工具 | [test_profiler.py](ut/tools/test_profiler.py) |
| **Data** | 数据工具函数 | [test_utils.py](ut/data/data_utils/test_utils.py) |
| | 多模态数据处理插件 | [test_mm_plugin.py](ut/data/data_utils/func_utils/test_mm_plugin.py) |
| **Models - Vision** | Vision RoPE索引计算 (Qwen2VL) | [test_qwen2vl_get_rope_index.py](ut/models/vision/test_qwen2vl_get_rope_index.py) |
| | Vision RoPE索引计算 (Qwen2.5VL) | [test_qwen2_5vl_get_rope_index.py](ut/models/vision/test_qwen2_5vl_get_rope_index.py) |
| | Vision RoPE索引计算 (Qwen2.5Omni) | [test_qwen2_5_omni_get_rope_index.py](ut/models/vision/test_qwen2_5_omni_get_rope_index.py) |
| | Vision RoPE Processor (Qwen2VL) | [test_qwen2vl_rope_processor.py](ut/models/vision/vision_encoders/test_qwen2vl_rope_processor.py) |
| **Models - Transformers** | Attention Utils (Qwen3VL) | [test_attention_utils.py](ut/models/transformers/qwen3vl/test_attention_utils.py) |
| | Attention Modules (Qwen3Omni) | [test_attention_modules.py](ut/models/transformers/qwen3omni/test_attention_modules.py) |
| **Models - Text Encoder** | 文本编码器处理 | [test_text_encoder_processor.py](ut/models/text_encoder/test_text_encoder_processor.py) |
| | Tokenizer处理 | [test_tokenzier_processor.py](ut/models/text_encoder/test_tokenzier_processor.py) |
| **Models - Audio Encoder** | 音频编码器处理 | [test_audio_encoder_processor.py](ut/models/audio_encoder/test_audio_encoder_processor.py) |
| **Models - AE** | AutoEncoder处理 | [test_ae_processor.py](ut/models/ae/test_ae_processor.py) |
| **Models - Diffusion** | IDDPM Scheduler | [test_iddpm.py](ut/models/diffusion/test_iddpm.py) |
| | Diffusers Scheduler | [test_diffusers_scheduler.py](ut/models/diffusion/test_diffusers_scheduler.py) |
| | Wan Flow Match Scheduler | [test_wan_flow_match_scheduler.py](ut/models/diffusion/test_wan_flow_match_scheduler.py) |
| | CogVideoX扩散模型 | [test_cogvideo_diffusion.py](ut/models/diffusion/test_cogvideo_diffusion.py) |
| | Hunyuan I2V扩散模型 | [test_hunyuan_i2v_diffusion.py](ut/models/diffusion/test_hunyuan_i2v_diffusion.py) |
| **Models - Common** | 激活函数 | [test_activations.py](ut/models/common/test_activations.py) |
| | 注意力机制 | [test_attention.py](ut/models/common/test_attention.py) |
| | 非对齐分割 | [test_unaligned_split.py](ut/models/common/test_unaligned_split.py) |
| | 位置编码 | [test_pos_embeddings.py](ut/models/common/embeddings/test_pos_embeddings.py) |
| | CogVideoX位置编码 | [test_cogvideox_pos_emb.py](ut/models/common/embeddings/test_cogvideox_pos_emb.py) |
| **Tasks** | Sora GRPO Trainer | [test_sora_grpo_trainer.py](ut/tasks/dancegrpo/test_sora_grpo_trainer.py) |
| | Flux GRPO Trainer | [test_flux_grpo_trainer.py](ut/tasks/dancegrpo/test_flux_grpo_trainer.py) |
| **FSDP** | Chunk Gated Delta Rule (Qwen3.5) | [test_chunk_gated_delta_rule.py](ut/fsdp/models/qwen3_5/test_chunk_gated_delta_rule.py) |
| **Checkpoint** | 权重转换 | [test_weight_convert.py](ut/test_weight_convert.py) |
| | Encoder Balance Comm | [test_encoder_balance_comm.py](ut/test_encoder_balance_comm.py) |
| | MoE Expert Weight Convert | [test_moe_expert_weight_convert.py](ut/test_moe_expert_weight_convert.py) |

---

## 三、开发流程

```mermaid
flowchart LR
    A[需求分析] --> B[用例设计]
    B --> C[代码开发]
    C --> D[本地验证]
    D --> E[CI门禁]
    E --> F[PR评审]
    F --> G[合入代码]
```

---

## 四、开发规范

### 4.1 命名规范

#### 4.1.1 ST用例命名规则

| 测试类型 | 命名规则 | 示例 |
| :--------: | :--------- | :----- |
| pretrain | `pretrain_` + 模型名 + `.sh` | `pretrain_cogvideox_t2v_1_0.sh` |
| finetune | `finetune_` + 模型名 + `.sh` | `finetune_qwen2vl_7B.sh` |
| posttrain | `posttrain_` + 模型名 + `_` + 任务类型 + `.sh` | `posttrain_qwen2vl_dpo.sh` |
| inference | `inference_` + 模型名 + `.sh` | `inference_qwen2vl_7b_pp1.sh` |

#### 4.1.2 UT用例命名规则

```text
test_ + 目标文件名或特性、功能名
```

**示例**：`test_chunkloss.py`

### 4.2 用例规范

#### 4.2.1 ST用例要求

1. **环境配置**：因为CI服务器硬件是NPU，必须设置正确的NPU环境变量
2. **数据shuffle必须关闭**：多模态训练用例中需关闭数据shuffle以确保结果可复现
3. **模型减层运行**：为节省资源同时保证测试有效性，模型需要减层运行，但层数不能设置过低以避免性能波动过大
4. **基线数据**：每个ST用例需配套基线数据文件，放置于 `st/baseline_results/` 目录，文件名为 `${script_name}.json`

#### 4.2.2 UT用例要求

1. **代码编写风格**：需与现有UT用例保持一致
2. **命名规范**：所有用例以 `test` 作为命名前缀
3. **目录层级**：建议按照功能特性进行文件夹命名区分

#### 4.2.3 CI门禁时间要求

- 整个CI门禁执行时间须**小于40分钟**

#### 4.2.4 资源路径规范

| 资源类型 | 路径 |
| ---------- | ------ |
| 模型权重 | `/home/ci_resource/models` |
| 数据集 | `/home/ci_resource/data` |

---

## 五、附录

### 5.1 目录结构说明

```text
tests/
├── README.md                        # 本文档
├── conftest.py                      # pytest全局配置
├── st/                              # 系统测试用例
│   ├── shell_scripts/               # ST脚本存放目录
│   │   ├── pretrain_*.sh            # 预训练用例
│   │   ├── finetune_*.sh            # 微调用例
│   │   ├── posttrain_*.sh           # 后训练用例
│   │   └── inference_*.sh           # 推理用例
│   ├── run_configs/                 # 用例配置文件目录
│   ├── baseline_results/            # 基线数据目录
│   ├── st_run.sh                    # ST用例执行入口
│   └── local_st_run.sh              # 本地ST执行脚本
└── ut/                              # 单元测试用例
    ├── loss/                        # Loss相关UT
    ├── tools/                       # 工具相关UT
    ├── data/                        # 数据处理UT
    ├── models/                      # 模型相关UT
    │   ├── vision/                  # 视觉模型UT
    │   ├── transformers/            # Transformer UT
    │   ├── text_encoder/            # 文本编码器UT
    │   ├── audio_encoder/           # 音频编码器UT
    │   ├── ae/                      # 自编码器UT
    │   ├── diffusion/               # 扩散模型UT
    │   └── common/                  # 通用模块UT
    ├── tasks/                       # 任务相关UT
    ├── tools/                       # 工具UT
    ├── fsdp/                        # FSDP相关UT
    └── test_*.py                    # 根目录UT
```
