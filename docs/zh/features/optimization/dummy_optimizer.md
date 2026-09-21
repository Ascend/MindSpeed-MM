# Dummy Optimizer

## 适用后端

MCORE

## 问题分析

朴素的pipeline parallel实现中，不支持某个pipeline stage的parameter都不需要参数更新或不需要反向计算。

## 解决方案

创建空tensor，规避optimizer中所有parameter都不需要更新的场景。
在pipeline parallel的反向前加判断，若没有grad_fn则不进行反向计算。

## 使用方法

启用前提：典型场景为pipeline parallel下某stage的参数全部冻结、导致该stage没有梯度（可参考 `examples/qwen2.5vl/README.md` 中的说明）。

1. 在模型入口脚本中导入patch模块：

   ```python
   from mindspeed_mm.patchs import dummy_optimizer_patch
   ```

2. 在模型启动shell中添加参数：

   ```shell
   GPT_ARGS="
       ...
       --enable-dummy-optimizer \
   "
   ```

## 注意事项

### 已知约束

- 仅 `pretrain_vlm.py`、`posttrain_sora_dpo.py` 入口已导入patch模块；其余入口（`pretrain_transformers.py`、`pretrain_sora.py`、`pretrain_omni.py`、`posttrain_flux_dancegrpo.py`）均未导入，这些入口下传参不生效。
