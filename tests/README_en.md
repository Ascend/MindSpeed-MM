# MindSpeed MM Test Case Writing Guide

This document describes in detail how to contribute DT test cases to MindSpeed MM.

## 1. Background and References

### 1.1 Code-Related Paths for CI Gate in the MindSpeed MM Repository

| Purpose | Path |
| ------ | ------ |
| Test cases | `MindSpeed-MM/tests` |
| CI startup code | `MindSpeed-MM/ci` |

### 1.2 CI Gate Scope

The CI gate guards the following two metrics:

1. **Functionality**: The code runs normally.
2. **Performance**: Performance degradation must not exceed 5%.

---

## 2. CI Gate Watch List

All PRs must pass the full CI gate test cases before being merged.

### 2.1 ST Watch List

> **Note**: System Test (ST) cases guard performance metrics, and performance degradation must not exceed 5%.

| Module | Features | Scripts |
| :------- | :--------- | :-------- |
| **Pretrain** | Wan2.1 T2V, FSDP2 | [pretrain_wan2.1_t2v.sh](st/shell_scripts/pretrain_wan2.1_t2v.sh) |
| | Wan2.2 I2V, FSDP2 | [pretrain_wan2.2_i2v.sh](st/shell_scripts/pretrain_wan2.2_i2v.sh) |
| **Finetune** | Qwen2.5VL 7B, TP=2, PP=2 | [finetune_qwen2_5_vl_7b.sh](st/shell_scripts/finetune_qwen2_5_vl_7b.sh) |
| | Qwen3Omni, FSDP2 | [finetune_qwen3omni.sh](st/shell_scripts/finetune_qwen3omni.sh) |
| | Qwen3VL 30B, FSDP2 | [finetune_qwen3vl_30B.sh](st/shell_scripts/finetune_qwen3vl_30B.sh) |
| | Kimi-K2.5, FSDP2 | [finetune_kimik2_5.sh](st/shell_scripts/finetune_kimik2_5.sh) |
| **Inference** | Wan2.2 T2V, CP=2 | [inference_wan2.2_t2v.sh](st/shell_scripts/inference_wan2.2_t2v.sh) |

### 2.2 UT Watch List

> **Note**: Unit Test (UT) cases guard functional metrics to ensure that the code runs correctly.

| Module | Features | Scripts |
| :------- | :--------- | :-------- |
| **Loss** | Chunk loss | [test_chunkloss.py](ut/loss/test_chunkloss.py) |
| **Tools** | Profiler analysis tool | [test_profiler.py](ut/tools/test_profiler.py) |
| **Data** | Data utility functions | [test_utils.py](ut/data/data_utils/test_utils.py) |
| | Multimodal data processing plugin | [test_mm_plugin.py](ut/data/data_utils/func_utils/test_mm_plugin.py) |
| **Models - Vision** | Vision RoPE index calculation (Qwen2VL) | [test_qwen2vl_get_rope_index.py](ut/models/vision/test_qwen2vl_get_rope_index.py) |
| | Vision RoPE index calculation (Qwen2.5VL) | [test_qwen2_5vl_get_rope_index.py](ut/models/vision/test_qwen2_5vl_get_rope_index.py) |
| | Vision RoPE index calculation (Qwen2.5Omni) | [test_qwen2_5_omni_get_rope_index.py](ut/models/vision/test_qwen2_5_omni_get_rope_index.py) |
| | Vision RoPE Processor (Qwen2VL) | [test_qwen2vl_rope_processor.py](ut/models/vision/vision_encoders/test_qwen2vl_rope_processor.py) |
| **Models - Text Encoder** | Text encoder processing | [test_text_encoder_processor.py](ut/models/text_encoder/test_text_encoder_processor.py) |
| | Tokenizer processing | [test_tokenzier_processor.py](ut/models/text_encoder/test_tokenzier_processor.py) |
| **Models - Audio Encoder** | Audio encoder processing | [test_audio_encoder_processor.py](ut/models/audio_encoder/test_audio_encoder_processor.py) |
| **Models - AE** | AutoEncoder processing | [test_ae_processor.py](ut/models/ae/test_ae_processor.py) |
| **Models - Diffusion** | Wan Flow Match Scheduler | [test_wan_flow_match_scheduler.py](ut/models/diffusion/test_wan_flow_match_scheduler.py) |
| **Models - Common** | Activation functions | [test_activations.py](ut/models/common/test_activations.py) |
| | Unaligned split | [test_unaligned_split.py](ut/models/common/test_unaligned_split.py) |
| | Positional encoding | [test_pos_embeddings.py](ut/models/common/embeddings/test_pos_embeddings.py) |
| | CogVideoX positional encoding | [test_cogvideox_pos_emb.py](ut/models/common/embeddings/test_cogvideox_pos_emb.py) |
| **Tasks** | Sora GRPO Trainer | [test_sora_grpo_trainer.py](ut/tasks/dancegrpo/test_sora_grpo_trainer.py) |
| | Flux GRPO Trainer | [test_flux_grpo_trainer.py](ut/tasks/dancegrpo/test_flux_grpo_trainer.py) |
| **Checkpoint** | Weight conversion | [test_weight_convert.py](ut/test_weight_convert.py) |
| | Encoder Balance Comm | [test_encoder_balance_comm.py](ut/test_encoder_balance_comm.py) |
| | MoE Expert Weight Convert | [test_moe_expert_weight_convert.py](ut/test_moe_expert_weight_convert.py) |

---

## 3. Development Process

```mermaid
flowchart LR
    A[Requirement Analysis] --> B[Test Case Design]
    B --> C[Code Development]
    C --> D[Local Verification]
    D --> E[CI Gate]
    E --> F[PR Review]
    F --> G[Code Merging]
```

---

## 4. Development Standards

### 4.1 Naming Conventions

#### 4.1.1 ST Case Naming Rules

| Test Type | Naming Rules | Example |
| :--------: | :--------- | :----- |
| pretrain | `pretrain_` + Model Name + `.sh` | `pretrain_cogvideox_t2v_1_0.sh` |
| finetune | `finetune_` + Model Name + `.sh` | `finetune_qwen2vl_7B.sh` |
| posttrain | `posttrain_` + Model Name + `_` + Task Type + `.sh` | `posttrain_qwen2vl_dpo.sh` |
| inference | `inference_` + Model Name + `.sh` | `inference_qwen2vl_7b_pp1.sh` |

#### 4.1.2 UT Case Naming Rules

```text
test_ + Target_file/feature/function_name
```

**Example**: `test_chunkloss.py`

### 4.2 Test Case Specifications

#### 4.2.1 ST Case Requirements

1. **Environment configuration**: Because the CI server hardware is an NPU, the correct NPU environment variables must be set.
2. **Data shuffle must be disabled**: In multimodal training test cases, data shuffle must be disabled to ensure reproducible results.
3. **Model runs with reduced layers**: To save resources while ensuring test validity, the model needs to run with reduced layers, but the number of layers must not be set too low to avoid excessive performance fluctuation.
4. **Baseline data**: Each ST case must be accompanied by a baseline data file, placed in the `st/baseline_results/` directory, with the file name `${script_name}.json`.

#### 4.2.2 UT Case Requirements

1. **Code style**: Must be consistent with existing UT cases.
2. **Naming convention**: All test cases must use `test` as the naming prefix.
3. **Directory hierarchy**: It is recommended to name folders by functional feature for distinction.

#### 4.2.3 CI Gate Time Requirements

- The total CI gate execution time **must be less than 40 minutes**.

#### 4.2.4 Resource Path Specifications

| Resource Type | Path |
| ---------- | ------ |
| Model weights | `/home/ci_resource/models` |
| Dataset | `/home/ci_resource/data` |

---

## 5. Appendix

### 5.1 Directory Structure

```text
tests/
├── README.md                        # This document
├── conftest.py                      # pytest global configuration
├── st/                              # System test cases
│   ├── shell_scripts/               # ST script directory
│   │   ├── pretrain_*.sh            # Pretraining test cases
│   │   ├── finetune_*.sh            # Fine-tuning test cases
│   │   ├── posttrain_*.sh           # Post-training test cases
│   │   └── inference_*.sh           # Inference test cases
│   ├── run_configs/                 # Test case configuration file directory
│   ├── baseline_results/            # Baseline data directory
│   ├── st_run.sh                    # ST test case execution entry point
│   └── local_st_run.sh              # Local ST execution script
└── ut/                              # Unit test cases
    ├── loss/                        # Loss-related UT
    ├── tools/                       # Tool-related UT
    ├── data/                        # Data processing UT
    ├── models/                      # Model-related UT
    │   ├── vision/                  # Vision model UT
    │   ├── transformers/            # Transformer UT
    │   ├── text_encoder/            # Text encoder UT
    │   ├── audio_encoder/           # Audio encoder UT
    │   ├── ae/                      # Autoencoder UT
    │   ├── diffusion/               # Diffusion model UT
    │   └── common/                  # Common module UT
    ├── tasks/                       # Task-related UT
    ├── tools/                       # Tool UT
    ├── fsdp/                        # FSDP-related UT
    └── test_*.py                    # Root directory UT
```
