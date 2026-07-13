# Hetero-Parallel

## Technical Background

The current mainstream training paradigm based on Megatron (primarily DP/PP/TP, supplemented by SP/CP/EP) is suitable for homogeneous models like LLMs (a single model with a regular, structured layer arrangement where each layer or a few layers can be seen as a repeating basic unit). Multimodal large models (MLLMs, Omni) are heterogeneous models typically composed of multiple modality encoders, a backbone network, and decoders, with significant differences in model architecture. When using Megatron-LM to train multimodal models, the MLLM is treated as a whole, with the encoder, LLM, and generator all using the same DP, TP, and other distributed strategies. The encoder and generator, as additional PP stages, are integrated into the backbone LLM, which leads to two types of problems: model heterogeneity and data heterogeneity.

- Model Heterogeneity: Different encoders (vision/audio) and the backbone (LLM) have varying computational loads and model sizes, leading to memory and computation imbalances that cause computational bubbles. This specifically manifests as `LLM bound` and `encoder bound` phenomena.
- Data Heterogeneity: The number of tokens for different modalities (text, speech, vision) in training samples varies greatly and changes dynamically (e.g., dynamic resolution). This leads to variations in the computational load across different encoders and the backbone, causing load imbalance. Specific issues include `intra-microbatch` and `inter-microbatch` imbalance.

To address model heterogeneity and data heterogeneity, MindSpeed MM has designed `hetero-parallel` and an [online data rearrangement scheme](./online_data_rearrange.md), respectively.

## hetero-parallel

### Solution Introduction

`hetero-parallel` (heterogeneous parallelism) decouples the parallel configuration of multimodal models, allowing each submodule to independently configure its own parallelism strategies, thereby resolving the imbalance between computation and storage. Unlike the `dist-train` approach, `hetero-parallel` employs a hybrid deployment of encoders and the backbone network, eliminating the computation bubbles and resource waste caused by independent deployment.

A brief description of the implementation is as follows:

- Online `parallel_state` converter: Stores snapshots of the runtime configurations for different submodules. It dynamically modifies the mpu (model parallel unit) state of the currently running module at runtime, enabling dynamic switching of parallel configurations.
- Data distribution utility: A module for the data flow from the encoder to the LLM, ensuring data flow correctness. Communication overlap is currently being implemented.
- Model hooks: Hooks mounted before and after the model's forward and backward passes to handle data flow conversion for different modules and to switch mpu states.
- Heterogeneous PP: Implements the `forward_backward_func_list` scheduling.

Two recommended usage scenarios are:

#### Heterogeneous DP/TP/CP

Given current model loads (such as the QwenVL series), the encoder is typically small with low static memory overhead, while the LLM is large with high static overhead. For such scenarios, the encoder can use DP/CP, and the LLM can use DP/TP/CP. For example, for a short-sequence scenario on the Qwen2.5Omni 7B model, configuring ViT and Audio with DP8, and the LLM with TP4DP2 yields optimal performance.

#### Heterogeneous PP

For scenarios requiring PP (small MBS, large GAS, and a large number of LLM parameters), heterogeneous PP can be used. The ViT and Audio encoders utilize large DP, while the LLM enables PP. It supports using different MBS for the encoder and LLM. It is recommended to use a larger MBS for the encoder (encoder MBS ~= 4-8 LLM MBS) to achieve better performance.

### How to Enable

1. Add the following parameters to the training launch script.

    ```shell
    GPT_ARGS="
        ...
        --hetero-parallel \
        --hetero-encoder-mbs-scale {num} \   # Adjust the MBS of the image/audio encoder to be num times that of the text decoder to improve computational efficiency
    "
    ```

2. Add parameters such as `tp/pp/cp` and `mbs` to the submodules requiring `hetero-parallel` in the corresponding `model.json`. Note that the backbone network no longer supports initializing parallel strategies via shell scripts, and all parallel strategies in the shell script must be set to 1.

    ```txt
    {
       ...
        "image_encoder": {
            "vision_encoder": {
                ...
                "tp":1,
                "pp":1,
                "cp":1
           },
        },
       "audio_encoder": {
            ...
            "tp":1,
            "pp":1,
            "cp":1
        },
        "text_decoder": {
            ...
            "tp":1,
            "pp":1,
            "cp":1
        },
       ...
    }
    ```

    ```shell
    TP=1
    PP=1
    CP=1
    ```

### Scope of Application

Currently, only models trained with pretrain_vlm using the Megatron backend, instead of the FSDP2 backend, support this feature.
