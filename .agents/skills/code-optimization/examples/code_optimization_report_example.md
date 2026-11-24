# MindSpeed-MM 代码优化候选报告（示例）

本报告记录使用新版 `code-optimization` 技能对 MindSpeed-MM 进行的一次静态扫描。候选补丁由评审人逐条决定是否采纳。

- 生成日期：2026-11-20。
- 扫描基线：`upstream/master` 的 `84734f18`。
- 扫描范围：`mindspeed_mm/` 下 714 个 Python 文件。使用 AST 核查重复定义和重复导入，用 pyflakes 筛查未用绑定，再按实际调用路径判断等价性。
- backend 语境：同时扫描 Megatron 与 FSDP2。提交的候选均位于 Megatron 侧，FSDP2 算子命中列入跳过项。
- 验证：8 个补丁均在扫描基线上通过 `git apply --check` 并在临时目录应用。7 个受影响文件通过 `py_compile`，pyflakes 告警从 22 条降到 12 条。`co-3` 使用 `--ignore-whitespace` 处理 CRLF 行尾。
- 验证范围：保语义判断基于静态证据。NPU 训练和分布式回归测试请在采纳后按改动范围执行。
- 声明：仓库源码未修改。

## 1. 候选清单

| ID | 文件 | 行 | 类别 | value | tier | ripple_complete | 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| co-1 | `mindspeed_mm/models/predictor/dits/sat_dit.py` | 845-848 | dedup | 2 | 可以改 | true | 待定 |
| co-2 | `mindspeed_mm/models/common/communications.py` | 38-83 | deadcode | 2 | 可以改 | true | 待定 |
| co-3 | `mindspeed_mm/models/predictor/dits/step_video_dit.py` | 30 | import | 1 | 可以改 | true | 待定 |
| co-4 | `mindspeed_mm/models/common/module_spec/glm4v_layer_spec.py` | 37-39 | import | 1 | 可以改 | true | 待定 |
| co-5 | `mindspeed_mm/models/vision/vision_encoders/qwen3vl_vit_model.py` | 36 | import | 1 | 可以改 | true | 待定 |
| co-6 | `mindspeed_mm/models/predictor/dits/sat_dit.py` | 290 | deadcode | 1 | 可以改 | true | 待定 |
| co-7 | `mindspeed_mm/utils/utils.py` | 661 | deadcode | 1 | 可以小改 | true | 待定 |
| co-9 | `mindspeed_mm/mcore/models/qwen3_5/modeling_qwen3_5.py` | 30 | import | 1 | 可以改 | true | 待定 |

## 2. 候选详情

### co-1

- file：`mindspeed_mm/models/predictor/dits/sat_dit.py`
- lines：845-848
- category：dedup
- value：2
- tier：可以改
- ripple_complete：true
- problem：`VideoDiTBlock.set_chunk_feed_forward` 在第 845 和 993 行各定义一次。
- rationale：两份定义的签名、默认值和函数体相同，均无装饰器。移除第一份后，类上继续使用第 993 行的实现。
- ripple：全仓追踪 `set_chunk_feed_forward` 与 `VideoDiTBlock`，另外的同名定义属于其他类，未发现该类的子类或对此方法的调用。现有方法签名保留。

```diff
--- a/mindspeed_mm/models/predictor/dits/sat_dit.py
+++ b/mindspeed_mm/models/predictor/dits/sat_dit.py
@@ -842,10 +842,6 @@

         return latents

-    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
-        self._chunk_size = chunk_size
-        self._chunk_dim = dim
-
     def before_attention(
         self,
         latents: torch.Tensor,
```

### co-2

- file：`mindspeed_mm/models/common/communications.py`
- lines：38-83
- category：deadcode
- value：2
- tier：可以改
- ripple_complete：true
- problem：`_adjust_tensor_dimensions` 和 `_unadjust_tensor_dimensions` 是两段未被引用的 Megatron 侧私有函数。
- rationale：两份定义无装饰器和默认参数，定义时无需运行函数体。删除它们保留文件中实际执行的通信调用。
- ripple：全仓搜索两个带下划线的名字只命中定义。FSDP2 侧同类函数独立定义并有调用方。未发现这两个私有名字的导出或动态访问。

```diff
--- a/mindspeed_mm/models/common/communications.py
+++ b/mindspeed_mm/models/common/communications.py
@@ -33,52 +33,6 @@
     split_forward_gather_backward_with_megatron_cp,
     split_forward_gather_backward_with_megatron_cp_tnd
 )
-
-
-def _adjust_tensor_dimensions(tensor, scatter_idx, gather_idx):
-    """
-    Adjust the dimensions of a tensor to move scatter_idx and gather idx to dim 0 and dim 1 respectively.
-    """
-    dims = list(range(tensor.dim()))
-
-    if gather_idx == 0:
-
-        if scatter_idx != 1:
-            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
-            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
-        # scatter_idx == 1:
-        else:
-            dims[scatter_idx], dims[gather_idx] = dims[gather_idx], dims[scatter_idx]
-
-    elif gather_idx == 1:
-        # scatter idx >= 2
-        if scatter_idx != 0:
-            # if scatter_idx is not 0, move it to 0
-            dims[0], dims[scatter_idx] = dims[gather_idx], dims[0]
-
-    # Handle the case when gather_idx >= 2
-    else:
-        if scatter_idx == 0:
-            dims[1], dims[gather_idx] = dims[scatter_idx], dims[0]
-
-        else:
-            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
-            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
-
-    return tensor.permute(dims).contiguous(), dims
-
-
-def _unadjust_tensor_dimensions(tensor, adjusted_dims):
-    """
-    Reverses the dimension adjustments using the list if adjusted dimensions.
-    """
-    inverse_dims = [0] * len(adjusted_dims)
-    for new_pos, old_pos in enumerate(adjusted_dims):
-        inverse_dims[old_pos] = new_pos
-
-    # Restore the dimension order
-    unadjusted_tensor = tensor.permute(inverse_dims).contiguous()
-    return unadjusted_tensor


 def cal_split_sizes(dim_size: int, world_size: int):
```

### co-3

- file：`mindspeed_mm/models/predictor/dits/step_video_dit.py`
- lines：30
- category：import
- value：1
- tier：可以改
- ripple_complete：true
- problem：第 27、30 行重复从 `einops` 导入 `rearrange` 和 `repeat`。
- rationale：保留第 27 行的导入。中间的两条 PyTorch 导入不会重新绑定这两个局部模块名。
- ripple：这两个名字仍从同一模块进入相同命名空间，文件内使用无需配套改动。原文件使用 CRLF，示例补丁采用 LF，验证与应用时指定 `--ignore-whitespace`。

```diff
--- a/mindspeed_mm/models/predictor/dits/step_video_dit.py
+++ b/mindspeed_mm/models/predictor/dits/step_video_dit.py
@@ -27,7 +27,6 @@
 from einops import rearrange, repeat
 import torch
 import torch.nn as nn
-from einops import rearrange, repeat
 from diffusers.models.embeddings import (
     PixArtAlphaTextProjection,
     TimestepEmbedding,
```

### co-4

- file：`mindspeed_mm/models/common/module_spec/glm4v_layer_spec.py`
- lines：37-39
- category：import
- value：1
- tier：可以改
- ripple_complete：true
- problem：`AttnMaskType`、`IdentityOp`、`ModuleSpec` 在第 23、24、26 行已从相同模块导入，第 37～39 行重复导入。
- rationale：保留先前的绑定和其他并列导入项，三个名字在中间没有被重新赋值。
- ripple：核对文件中的类型和构造路径，三个符号仍指向同一来源。导入该模块的调用方保留现有可访问名称。

```diff
--- a/mindspeed_mm/models/common/module_spec/glm4v_layer_spec.py
+++ b/mindspeed_mm/models/common/module_spec/glm4v_layer_spec.py
@@ -34,9 +34,6 @@
     TENorm,
 )
 from megatron.core.transformer.dot_product_attention import DotProductAttention
-from megatron.core.transformer.enums import AttnMaskType
-from megatron.core.transformer.identity_op import IdentityOp
-from megatron.core.transformer.spec_utils import ModuleSpec
 from megatron.core.transformer.transformer_layer import (
     TransformerLayer,
     TransformerLayerSubmodules,
```

### co-5

- file：`mindspeed_mm/models/vision/vision_encoders/qwen3vl_vit_model.py`
- lines：36
- category：import
- value：1
- tier：可以改
- ripple_complete：true
- problem：第 31 和 36 行从同一模块导入 `TransformerConfig`。
- rationale：第 31 行已位于 `try / except` 之外，保留该导入即可覆盖两条路径。
- ripple：后续配置类型仍由先前导入提供，文件内的引用与模块对外可见的名称保持一致。

```diff
--- a/mindspeed_mm/models/vision/vision_encoders/qwen3vl_vit_model.py
+++ b/mindspeed_mm/models/vision/vision_encoders/qwen3vl_vit_model.py
@@ -33,6 +33,5 @@

 from megatron.core import mpu
 from megatron.core.transformer.spec_utils import ModuleSpec
-from megatron.core.transformer.transformer_config import TransformerConfig
 from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region, gather_from_sequence_parallel_region
 from megatron.training import get_args
```

### co-6

- file：`mindspeed_mm/models/predictor/dits/sat_dit.py`
- lines：290
- category：deadcode
- value：1
- tier：可以改
- ripple_complete：true
- problem：`SatDiT.forward` 在第 290 行构造 `added_cond_kwargs`，此后没有读取它。
- rationale：右侧仅是字符串和 `None` 组成的字典字面量。删除局部赋值保留随后调用的参数和顺序。
- ripple：核对整个 `forward` 作用域，未发现该名字的读取。保留同名公开参数，调用方无需调整。

```diff
--- a/mindspeed_mm/models/predictor/dits/sat_dit.py
+++ b/mindspeed_mm/models/predictor/dits/sat_dit.py
@@ -287,7 +287,6 @@
             if "masked_video" in kwargs.keys() and kwargs["masked_video"] is not None:
                 latents = torch.cat([latents, kwargs["masked_video"]], dim=1)

-            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
             latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, \
                 embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
                 latents, prompt, timestep, frames)
```

### co-7

- file：`mindspeed_mm/utils/utils.py`
- lines：661
- category：deadcode
- value：1
- tier：可以小改
- ripple_complete：true
- problem：Megatron CP autograd 封装的 `backward` 读取 `ctx.cp_group`，但结果未被使用。
- rationale：正常入口通过 `apply` 创建上下文，`forward` 返回前在第 652 行保存该属性。移除的是既有上下文属性的读取。
- ripple：追踪 `split_forward_gather_backward_with_megatron_cp` 的 `apply` 调用与对应 `forward`。保留上下文写入、其余读取及所有通信和梯度运算，FSDP2 侧同名类有独立入口。

```diff
--- a/mindspeed_mm/utils/utils.py
+++ b/mindspeed_mm/utils/utils.py
@@ -658,7 +658,6 @@

     @staticmethod
     def backward(ctx, grad_output):
-        cp_group = ctx.cp_group
         cp_rank = ctx.cp_rank
         cp_size = ctx.cp_size
         seq_dim = ctx.seq_dim
```

### co-9

- file：`mindspeed_mm/mcore/models/qwen3_5/modeling_qwen3_5.py`
- lines：30
- category：import
- value：1
- tier：可以改
- ripple_complete：true
- problem：Qwen3.5 的 Megatron 实现于第 25 和 30 行重复导入 `PackedSeqParams`。
- rationale：保留第 25 行从相同模块导入的符号，中间没有对该符号重新赋值。
- ripple：核对第 241、678、684、771、893 行的类型标注与构造调用，它们均可通过第 25 行的绑定继续解析。模块导出名保持一致。

```diff
--- a/mindspeed_mm/mcore/models/qwen3_5/modeling_qwen3_5.py
+++ b/mindspeed_mm/mcore/models/qwen3_5/modeling_qwen3_5.py
@@ -27,7 +27,6 @@
 from megatron.core.transformer.enums import ModelType
 from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
 from megatron.core.transformer.module import MegatronModule
-from megatron.core.packed_seq_params import PackedSeqParams
 from megatron.core.transformer.spec_utils import ModuleSpec
 from megatron.core.inference.contexts import BaseInferenceContext
 from megatron.core.utils import deprecate_inference_params, WrappedTensor
```

## 3. 已识别但未提交

| 位置 | 现象 | 跳过原因 |
| --- | --- | --- |
| `mindspeed_mm/models/text_decoder/moe_model.py:588、637` | `MOEModel.sharded_state_dict` 两份定义 | 两份实现处理的 checkpoint 状态不同，需确认预期行为。 |
| `mindspeed_mm/models/predictor/dits/hunyuanvideo15/utils.py:182、476` | `auto_offload_model` 两份定义 | 参数默认值与模型为空时的处理不同，需确认预期行为。 |
| `mindspeed_mm/fsdp/ops/gdn/triton_core/utils.py:259、310` | `prepare_chunk_offsets` 的重复定义 | 涉及 Triton 算子与装饰器执行，超出本次可提候选范围。 |
| `mindspeed_mm/tasks/inference/pipeline/hunyuanvideo_15_pipeline.py:53、57` | 静态扫描命中同名 `num_timesteps` | 分别是属性 getter 与 setter。setter 提供参数校验，两个定义共同组成属性。 |
| `mindspeed_mm/models/common/communications.py:526` | `rank` 未读取，右侧调用 `torch.distributed.get_rank(process_group)` | 删除调用会改变非法进程组的报错时机。 |
| `mindspeed_mm/models/common/conv.py:242` | `req_send` 未读取 | 右侧调用 `isend` 发起通信，必须保留该调用。 |
| `mindspeed_mm/training.py:763、766` | `get_forward_backward_func` 条件替换 | 异构流水线通过这里选择实际的前向与反向执行函数。 |

## 4. 采纳后的落盘与验证

只应用评审选中的补丁，按改动范围运行 `pre-commit` 和相关测试。通信或训练链路的候选需要在对应的多卡环境验证。
