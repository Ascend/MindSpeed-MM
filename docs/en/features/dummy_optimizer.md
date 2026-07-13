# dummy optimizer

## Problem Analysis

In a naive pipeline parallel implementation, it is not supported that all parameters of a certain pipeline stage do not require parameter updates or backward computation.

## Solution

Create an empty tensor to avoid the scenario where all parameters in the optimizer do not require updates.
Add a check before the backward pass in a pipeline parallel implementation; if there is no `grad_fn`, skip the backward computation.

## How to Enable

1. Import the patch module in the model entry script (InternVL/Qwen2VL already supported).

   ```python
   from mindspeed_mm.patchs import dummy_optimizer_patch
   ```

2. Add the parameter in the model startup shell.

   ```shell
   GPT_ARGS="
       ...
       --enable-dummy-optimizer \
   "
   ```
