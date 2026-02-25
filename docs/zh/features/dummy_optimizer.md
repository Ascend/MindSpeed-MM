# dummy optimizer

## 问题分析

朴素的 pipeline parallel 实现中，不支持某个 pipeline stage 的 parameter 都不需要参数更新或不需要反向计算。

## 解决方案

创建空 tensor，规避 optimizer 中所有 parameter 都不需要更新的场景。
在 pipeline parallel 的反向前加判断，若没有 grad_fn 则不进行反向计算。

## 使用方法

1. 在模型入口脚本中导入 patch 模块（InternVL/Qwen2VL 已支持）；

   ```python
   from mindspeed_mm.patchs import dummy_optimizer_patch
   ```

2. 在模型启动 shell 中添加参数；

   ```shell
   GPT_ARGS="
       ...
       --enable-dummy-optimizer \
   "
   ```
