# parameter_lr_wd_tuning

## Problem Analysis

Multimodal large models involve the fusion of multiple modalities. For example, a VLM model includes various modules such as a Visual Encoder and a Language Decoder. Different modules (e.g., visual feature extraction layers, language attention layers, bias/normalization layers) exhibit varying sensitivities to Learning Rate (LR) and Weight Decay (WD). A globally uniform LR/WD configuration cannot accommodate the differentiated optimization needs of each module, potentially leading to underfitting or overfitting in some modules, which affects the model's multimodal alignment and final performance.

## Solution

A keyword-driven LR scaling and WD exclusion feature for parameter names is introduced:

1. Supports excluding WD for specified parameters by matching parameter names via keywords, preventing optimization bias caused by WD for specific parameters;
2. Supports applying a custom LR multiplier to specified parameters (e.g., the vision module of a VLM) by matching parameter names via keywords, enabling fine-grained parameter-level learning rate control.
3. All matching logic uses case-insensitive string matching.

## How to Enable

This feature is disabled by default. To enable it, add the following arguments to the model launch training script as need. Example:

   ```shell
   GPT_ARGS="
       ...
       --weight-decay-exclude-modules norm bias \ # Exclude weight decay for specified parameters, set parameter keywords based on actual needs
       --lr-scale-modules vision \  # Scale the learning rate for visual module parameters (e.g., mult=0.5)
       --lr-mult 0.5 \
   "
   ```
