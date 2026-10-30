# hetero-parallel

Last updated: 04/07/2026. Author: Shangda-Yang

## 1. Technical Background and Challenges

With the development of AI technologies, multimodal models (such as image-text and audio-video models) have become a research hotspot, and their scale is growing rapidly. These models usually contain multiple modules with different computational characteristics, such as Vision Transformer, Audio Encoder, and Text Decoder. Due to differences in model structure and data, these modules differ in efficiency under the same parallel strategy, making it difficult for a single parallel strategy to meet the training requirements of large-scale multimodal models.

### Challenges Faced by Existing Technologies

1. **Limitations of traditional parallel techniques**
   \- A uniform parallel strategy combination (DP/TP/PP/SP/CP) cannot fully exploit the computational characteristics of different modality modules
2. **Balance between memory and compute resources**
   \- Unbalanced allocation of compute and memory load leads to low resource utilization
   \- Local resource bottlenecks affect overall training performance

## 2. Overview

Hetero Parallel is a heterogeneous parallel technique for multimodal models in MindSpeed MM. It allows different modules of a model (such as the image encoder, audio encoder, and text decoder) to use different parallel configurations (for example, DP/TP/CP/SP/PP), so that compute resources can be used more flexibly and performance can be optimized.

Note: no scenario has been identified so far in which an encoder uses independent PP, so PP is not supported for the encoder part yet.

<!-- The current deployment scheme is fully co-located heterogeneous parallelism:

<img src="../../../../sources/images/features/hetero-parallel/fully_colocated_deployment_en.png" alt="Fully co-located deployment" width="50%">

Support for partially co-located heterogeneous parallelism is planned in the future.

<img src="../../../../sources/images/features/hetero-parallel/partially_colocated_deployment_en.png" alt="Partially co-located deployment" width="50%"> -->

## 3. Core Design

- **Module-level parallel configuration**: set an independent parallel strategy for different components of a multimodal model (image_encoder, audio_encoder, text_decoder)
- **Dynamic parallel state switching**: automatically switch the parallel state during the forward/backward pass of different modules
- **Efficient data transfer**: implement a mechanism for distributing and collecting data between different parallel groups
- **Flexible pipeline parallelism**: support heterogeneous pipeline parallelism (the text decoder part schedules its pipeline independently and flexibly reuses the existing pipeline scheduling strategies for large models) to optimize the training flow of multimodal models

## 4. Key Implementation Files

### 4.1 Core Implementation: `hetero_parallel.py`

Path: `MindSpeed-MM/mindspeed_mm/utils/hetero_parallel.py`

Main functions:

- Defines the list of supported heterogeneous parallel modules
- Implements the hooks that switch the parallel state before and after the model forward/backward pass
- Provides the core functions for transferring data between different parallel groups
- Manages the parallel state of different modules

### 4.2 Pipeline Parallel Patch: `hetero_pipeline_patches.py`

Path: `MindSpeed-MM/mindspeed_mm/patchs/hetero_pipeline_patches.py`

Main functions:

- Extends the standard pipeline parallel mechanism to support heterogeneous parallelism
- Implements the pipeline execution logic across multiple modules
- Manages the data flow between different modules

### 4.3 VLM Model Integration: `vlm_model.py`

Path: `MindSpeed-MM/mindspeed_mm/models/vlm_model.py`

Main functions:

- Integrates the hetero parallel capability into the multimodal model
- Initializes the parallel settings of different modules based on the configuration
- Manages parallel state switching between modules

## 5. Core Functionality Implementation

<!-- The following figure shows how the feature works:

<img src="../../../../sources/images/features/hetero-parallel/overall_workflow_en.png" alt="Overall workflow" width="50%"> -->

1. **Parallel initialization**: parse the model configuration and create an independent parallel state snapshot for each module
2. **Model initialization**: create the multimodal model and register the module forward hooks
3. **Forward pass**:
   \- The module forward pre-hook is triggered and switches to the parallel state of the corresponding module
   \- Collect or split data to fit the parallel configuration of the target module
   \- Execute the module forward computation
   \- The module backward hook is triggered and restores the original parallel state
4. **Pipeline execution**: in heterogeneous pipeline mode, multiple modules execute in sequence, achieving efficient pipeline parallelism

The implementation of the main modules is described in the subsections of this chapter.

### 5.1 Parallel State Management

```python
# List of supported heterogeneous parallel modules
_HeteroParallelModules = ['image_encoder', 'audio_encoder', 'text_decoder']

# Parallel state dictionary, used to store the parallel state snapshot of each module
_ParallelStatesDict = {}

# Initialize the parallel state of each module and save the snapshot
def initial_modules_mpu(reuse_module, args):
    # Initialize the module parallel state and save the snapshot

# Switch the parallel state
def change_parallel_state(module):
    # Switch the parallel state between different modules (module)
```

### 5.2 Module Forward Hooks

Register hook functions before and after the forward pass for different modules to implement automatic parallel state switching and data processing:

```python
# Pre-hook for the image encoder forward pass
def image_encoder_forward_pre_hook(module, input):
    # Implement parallel state switching and data collection

# Post-hook for the image encoder forward pass
def image_encoder_forward_hook(module, input, output):
    # Implement data collection and parallel state switching

# Pre-hook for the audio encoder forward pass
def audio_encoder_forward_pre_hook(module, input):
    # Implement parallel state switching and data collection

# Post-hook for the audio encoder forward pass
def audio_encoder_forward_hook(module, input, output):
    # Implement data collection and parallel state switching
```

### 5.3 Data Parallel Group Operations

```python
def all_gather_dp_group(tensor, pad_token_id=None, cat_dim=0, pad_dim=1, remove_padding=False):
    # Gather tensors within the data parallel group
    # Support padding and unpadding of sequences with different lengths

def split_tensor_dp_group(tensor, split_dim=0, pad_dim=1, chunk_seq_lens=None):
    # Split tensors within the data parallel group
    # Support splitting by the specified length
```

### 5.4 Heterogeneous Pipeline Parallelism

```python
def hetero_pipeline(pipeline_meta_list, forward_backward_func_list, *, ...):
    # Implement heterogeneous pipeline parallel execution across multiple modules
    # Manage data flow and parallel state switching between different modules
```

## 6. Configuration and Usage

### 6.1 Weight Sharding

#### xP (encoders) + TP2 (text decoder) + PP2 (text decoder) sharding

```bash
# qwen2.5omni 7b(hf2mm)
mm-convert  Qwen2_5_OmniConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-Omni-7B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-Omni-7B" \
  --cfg.parallel_config.llm_pp_layers [[14,14]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.audio_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 2 \
  --cfg.parallel_config.vit_tp_size 1 \
  --cfg.parallel_config.audio_tp_size 1

# qwen2.5vl 7b(hf2mm)
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[12,16]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 2 \
  --cfg.parallel_config.vit_tp_size 1
```

#### Corresponding Inverse Weight Conversion

```bash
# qwen2.5omni 7b(mm2hf)
mm-convert  Qwen2_5_OmniConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-Omni-7B" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-Omni-7B" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-Omni-7B" \
  --cfg.parallel_config.llm_pp_layers [14,14] \
  --cfg.parallel_config.vit_pp_layers [32] \
  --cfg.parallel_config.audio_pp_layers [32] \
  --cfg.parallel_config.tp_size 2 \
  --cfg.parallel_config.vit_tp_size 1 \
  --cfg.parallel_config.audio_tp_size 1

# qwen2.5vl 7b(mm2hf)
mm-convert  Qwen2_5_VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [12,16] \
  --cfg.parallel_config.vit_pp_layers [32] \
  --cfg.parallel_config.tp_size 2 \
  --cfg.parallel_config.vit_tp_size 1
```

### 6.2 Enabling Heterogeneous Parallelism

Enable it through the command-line argument:

```bash
--hetero-parallel
```

### 6.3 Module Parallel Configuration

**Training script configuration**: set all parallel configurations in the training script to `1`.

```bash
TP=1
PP=1
CP=1
```

**Model parallel configuration**: set independent parallel parameters for different encoders and decoders in (`model_xx.json`) (the default example uses a full DP configuration):

```json
{
  "image_encoder": {
    "vision_encoder":{
      ...,
      "pipeline_num_layers": [32],
      ...
    },
    "tp": 1,
    "pp": 1, # Currently only pp=1 is supported
    "cp": 1
  },
  "audio_decoder": {
    "audio_decoder":{
      ...,
      "pipeline_num_layers": [32],
      ...
    },
    "tp": 1,
    "pp": 1, # Currently only pp=1 is supported
    "cp": 1
  },
  "text_decoder": {
    ...,
    "pipeline_num_layers": [11, 17],
    ...,
    "tp":1,
    "pp":1,
    "cp":1
  },
  ...
}
```

### 6.4 Encoder Over-Computation

After PP is enabled for the text decoder, the encoder can use a large MBS while the text decoder uses a small MBS under the same GBS. Feature data is buffered so that different modules can compute efficiently and compute resources are fully utilized.

How to enable:

```bash
--hetero-encoder-mbs-scale 4 # encoder mbs is 4 times that of the decoder
```

## 7. Technical Advantages

1. **Optimized resource utilization**: select the optimal parallel strategy based on the computational characteristics of different modules
2. **Flexible scalability**: support adding independent parallel configurations for new modules
3. **Performance improvement**: improve the training efficiency of multimodal models through heterogeneous parallelism and pipeline optimization
4. **Good compatibility**: integrate seamlessly with existing parallel frameworks and are easy to use
5. **Scenario gain**: in typical scenarios, the feature delivers an end-to-end performance gain of more than 20% on the Qwen2.5omni model (compared with a uniform parallel strategy for the entire network).

## 8. Application Scenarios

- **Large-scale multimodal model training**: the feature currently supports the Qwen2.5omni and Qwen2.5vl models.
- **Resource-constrained environments**: scenarios that require flexible allocation of compute resources.
- **Performance optimization**: optimize the parallel strategy for specific modules.
