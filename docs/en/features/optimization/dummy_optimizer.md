# Dummy Optimizer

## Applicable Backends

Mcore

## Problem Analysis

In a naive pipeline parallel implementation, it is not supported that all parameters of a certain pipeline stage do not require parameter updates or backward computation.

## Solution

Create an empty tensor to avoid the scenario where all parameters in the optimizer do not require updates.
Add a check before the backward pass in pipeline parallelism; if there is no `grad_fn`, skip the backward computation.

## How to Enable

Prerequisites: the typical scenario is that all parameters of a pipeline stage are frozen, so that the stage has no gradients (see the description in `examples/qwen2.5vl/README.md`).

1. Import the patch module in the model entry script:

   ```python
   from mindspeed_mm.patchs import dummy_optimizer_patch
   ```

2. Add the parameter in the model startup shell:

   ```shell
   GPT_ARGS="
       ...
       --enable-dummy-optimizer \
   "
   ```

## Notes

### Known Constraints

- Only the `pretrain_vlm.py` and `posttrain_sora_dpo.py` entries have imported the patch module, while the other entries (`pretrain_transformers.py`, `pretrain_sora.py`, `pretrain_omni.py`, `posttrain_flux_dancegrpo.py`) have not, so this parameter does not take effect for those entries.
