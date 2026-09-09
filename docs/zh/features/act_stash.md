# Act Stash（激活暂存）

> Act Stash是 [Swap Core](swap_core.md) 底座之上的租户，与底座的交互只有 `put` / `pop` 两个接口；换入换出的物理行为由共享的 `swap_plan` 配置决定。底座的行为与约定见 [Swap Core](swap_core.md)，本文只描述Act Stash自身的配置与语义。

## 适用后端

FSDP2

## 背景与挑战

长序列/大模型训练中，激活显存是主要瓶颈之一。即使开启重计算，checkpoint边界输入等由autograd保存的张量仍驻留HBM直至反向，层数与序列长度增长时这部分占用可观，削弱了重计算的显存收益。

## 解决方案

Act Stash是激活offload的 `saved_tensors_hooks` 实现（`activation_offload_plan.impl: "stash"`）：pack即 `put`、unpack即 `pop`——前向保存激活时换出、反向按需取回，租户不触碰换入换出的任何物理细节。它是仓库已有逐层激活offload（legacy实现，即 [Async Activation Offload](./async_activation_offload.md)）的另一种形态，两者二选一。取回miss一律直接报错：autograd保证每个packed张量必被unpack，miss恒为异常，不产生错误数值；反向取回序 ≈ 前向换出的逆序，与底座默认策略天然契合，通常无需为此调参。

### 覆盖范围与资格过滤

- **覆盖范围**：与non-reentrant checkpoint同开时，覆盖范围 = checkpoint边界输入 + checkpoint外层autograd保存的张量，与Op Replay（checkpoint内算子输出）互补、不重复。**未开重计算时**，覆盖范围变为包装模块内autograd保存的全部张量，换出流量与锁页占用相应增大。
- **资格过滤**：仅设备（NPU/CUDA）张量、非Parameter（及其视图）、strided布局、非空storage的张量会被换出；CPU张量、模型参数、稀疏等非strided布局张量原样透传，不进入缓存。
- **kwargs盲区**：non-reentrant checkpoint只对顶层位置参数走 `save_for_backward`（打包钩子通道）；具名kwargs与容器内的张量被checkpoint frame按引用持有，驻留HBM直至反向。可用 `recompute_plan.flatten_inputs: true`（默认关）把完整 `(args, kwargs)` 输入树摊平为去重的位置参数叶子纳入覆盖——keyword-only参数、`**kwargs` 内容与容器（list/dict）内的张量一并覆盖；checkpoint内部重建原始调用结构后执行前向，语义保持（跨槽位互为别名的对象共享同一叶子）。

## 使用场景

- **该开**：已开重计算、但checkpoint边界输入等autograd保存的激活仍驻留HBM（层数与序列长度增长时占用可观）时；或未开重计算、需要把包装模块内autograd保存的全部张量换出时（换出流量与锁页占用相应增大，需相应规划 `capacity_mb`）；与模型级 `skip_*_recompute` 开关的交互见「已知约束」。
- **不必开**：`capacity_mb` 不小于换出总量时无显存收益（口径见「已知约束」）；短序列、激活压力小的场景。

## 使用方法

开启条件：`enable_activation_offload: true` 且 `activation_offload_plan.apply_modules` 已显式配置（默认 `None`）且 `impl` 为 `"stash"`；`swap_plan` 可省略（走默认值）。

最小配置示例：

```yaml
features:
  enable_activation_offload: true
  activation_offload_plan:
    impl: stash
    apply_modules:
      - model.language_model.layers.{*}
  swap_plan:
    capacity_mb: 1024   # 显存收益口径见「已知约束」
```

### 参数说明

| 配置项 | 类型 | 默认值 | 含义与取值约束 |
| --- | --- | --- | --- |
| `enable_activation_offload` | bool | false | 是否开启激活offload。 |
| `activation_offload_plan.apply_modules` | List[str] / null | None | 应用激活offload的模块（pattern匹配，如 `model.language_model.layers.{*}`）。默认 `None`，需显式配置。 |
| `activation_offload_plan.impl` | str | "legacy" | 实现方式，二选一：`"legacy"` = 逐层hidden_states换出（自管理流与预取启发式，即 [Async Activation Offload](./async_activation_offload.md) 的路径）；`"stash"` = Act Stash（`saved_tensors_hooks` 租户，物理行为全部由 `swap_plan` 决定）。非法值启动校验报错。 |
| `recompute_plan.flatten_inputs` | bool | false | 把checkpoint边界完整 `(args, kwargs)` 输入树摊平为去重的位置参数叶子，使kwargs与容器内张量纳入Act Stash覆盖；仅non-reentrant生效。语义见[覆盖范围与资格过滤](#覆盖范围与资格过滤)。 |

### `capacity_mb` 怎么定

换出按层分块发生，**chunk = 单层换出量**，开重计算时约等于每层checkpoint边界输入大小：`micro_batch × seq × hidden × 字节数`（bf16为2）。以下为Act Stash单租户口径；与Op Replay同开时共享同一 `swap_plan` 预算，容量按两者需求之和规划（上界为两者换出总量之和），见 [Swap Core](swap_core.md) 的租户清单与选型指引。有效区间与推荐口径：

- **下界 = 1个chunk**：低于1 chunk时，每次put须等自身刚发起的拷贝完成才能放行，换出与计算完全串行（每步白付 ≈ `层数 × 单块拷贝耗时`），不如不开。
- **完全掩盖条件**：`capacity_mb ≥ chunk × ⌈单块拷贝耗时 / 单层前向耗时⌉`——**向上取整**，容量按chunk粒度生效：比值1.2时需2个chunk，只配1.2个则按1个的窗口计、仍会等待。拷贝比一层计算快时（常见），**1个chunk即可完全掩盖**，再加大不改善step time。
- **上界 = 换出总量**（chunk × 覆盖层数），达到后步内零释放、无显存收益（见「已知约束」）；**显存收益 ≈ 换出总量 − `capacity_mb`**：在掩盖下界与总量之间，`capacity_mb` 每加1MB收益少1MB——按「刚好掩盖」取最小值即可。

**示例**（qwen3_5 4B，hidden=2560，bf16，seq 16K，micro_batch=1）：chunk ≈ 16384 × 2560 × 2B ≈ 80MiB；32层全开重计算时换出总量 ≈ 2.5GiB。若实测单块拷贝耗时小于单层前向耗时（常见），`capacity_mb` 取2个chunk（160MiB，留一份抖动余量）即可完全掩盖，显存收益 ≈ 2.3GiB。拷贝带宽建议在目标机器上实测确认。

**调优信号**：开启后step time相对不开时的增量接近0即已掩盖；若明显退化，先检查 `capacity_mb` 是否低于1个chunk。

## 注意事项

### 已知约束

- **与skip重计算算子的交互**：模型级开关 `skip_flash_attn_recompute` / `skip_gdn_recompute`（qwen3_5等模型）的输出仍由legacy卸载协议释放（不经Swap Core），Act Stash不覆盖这组算子。推荐配对：`skip_*_recompute: false`，由Op Replay通过Swap Core覆盖同一组算子。
- **与重计算组合要求 `use_reentrant: false`**：覆盖语义按non-reentrant checkpoint定义，reentrant组合不在已验证范围。
- **显存收益取决于硬上限**：要省显存需 `capacity_mb` 小于换出总量（缓存池物理峰值与释放时机的机制见 [Swap Core](swap_core.md) 配置机制图解）；`keep_rate` 只调节拷贝时机与取回命中率，与显存收益无关。
- **禁止原地改写**：经Act Stash换出的张量（含checkpoint边界输入；开启 `flatten_inputs` 后还包括摊平进来的kwargs与容器内张量）在反向取回前禁止in-place改写，违反不报错、静默产生错误梯度。契约详见 [Swap Core](swap_core.md) 使用约定；确需改写请先 `clone()`。
