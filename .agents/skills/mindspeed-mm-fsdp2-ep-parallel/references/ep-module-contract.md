# MoE 模块的 EP 接入契约

阶段 1c/1d 判定布局与契约、阶段 2 改造模块时读本文。

## Source anchors

| 锚点 | 文件 |
| --- | --- |
| `expert_parallelize_modules`、`get_experts_forward_fn_for_qwen`、`distribute_expert_weight` | `mindspeed_mm/fsdp/distributed/expert_parallel/expert_parallel.py` |
| `expert_fully_shard_modules` | `mindspeed_mm/fsdp/distributed/expert_parallel/expert_fully_shard_parallel.py` |
| `ep_forward`、`ep_mc2_forward`、`ep_allgather_forward` | `mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py` |
| `WanMoEExperts`（自定义 `ep_forward` 样板） | `mindspeed_mm/fsdp/models/wan2_2/modeling_wan2_2.py` |

**使用某节前，先确认该节依赖的锚点（上表的函数/类名）仍存在**（grep 即可）。锚点还在，说明该节描述的机制大概率仍然成立；锚点已改名或删除，说明框架在本文写成后已演进，该节结论可能失效——此时以当前代码为准重新核实，并提醒用户更新本文。锚点的作用是让文档过期变得**可检测**：文档不锁死框架演进，但过期时能被发现，而不是被沿用。

---

## 1. 默认 EP forward 的假设（分支 A/B 判定依据）

`expert_parallelize_modules` 替换专家模块 forward 时：模块自带 `ep_forward` 则绑定它；否则注入 `get_experts_forward_fn_for_qwen` 生成的默认实现。默认实现按 Qwen 系 MoE 编写——这是仓内 Qwen 系模型零模型改动接 EP 的原因，也是分支判定的基准。

对照三项（假设的具体形式以函数体为准，属易变层，不在此写死）：

| 对照项 | 默认实现的假设 | 偏离时的处理 |
| --- | --- | --- |
| 权重属性 | 直接读取 Qwen 风格的融合权重属性（gate_up / down 两个投影）与元数据属性（专家数、hidden 尺寸） | 仅属性名不同：在模块上补别名属性，代价最小（wan2_2 补 `hidden_size` 即此做法）。结构不同：走 B |
| 激活与门控 | dispatcher 默认 SwiGLU（`apply_activation` 在未传自定义激活时走 swiglu） | 其他激活（GELU 等）：走 B，经 `ep_dispatcher.ep_forward` 的 `activation` 参数传入自定义 Callable |
| 调用约定 | MoE block 以（隐藏状态、路由权重、路由索引）三个位置参数调用专家容器 | 调用方式不同：走 B，在 `ep_forward` 内适配 |

分支 B 的原则：**偏离点通过 dispatcher 既有参数表达，不重写 dispatcher。** dispatcher 承载的是 token 路由与通信逻辑，与专家的具体计算无关；把模型差异塞进 dispatcher 会让一个模型的特例污染所有模型共用的路径。

`ep_forward` 内的 DTensor 处理：EP 分片后专家权重为 `DTensor`，计算前取 `to_local()`，并兼容未分片时的普通 Tensor。写法直接参照 `WanMoEExperts.ep_forward`。

---

## 2. 分支 C：per-expert 子模块 → 融合专家权重

EP 分片按张量第 0 维切且递归作用于容器下全部参数，`ModuleList` 形态无法表达「按专家维切分」，必须先融合。

以 Qwen 风格 per-expert 结构为例（示意代码，具体属性名与布局以仓内同类模型为准）：

**改造前**（HF 上游常见形态）：

```python
self.experts = nn.ModuleList([MLP(config) for _ in range(E)])
# 每个 MLP 含 gate_proj / up_proj / down_proj 三个 nn.Linear
```

**改造后**（融合形态，布局以 `[E, H, 2I]` / `[E, I, H]` 为例）：

```python
self.gate_up_proj = nn.Parameter(torch.empty(E, hidden, 2 * intermediate))
self.down_proj = nn.Parameter(torch.empty(E, intermediate, hidden))
```

**权重拼装**（加载路径同步改）。`nn.Linear.weight` 形状为 `[out, in]`，融合到 `[E, H, 2I]` 需转置：

```python
gate_up = torch.stack([
    torch.cat([m.gate_proj.weight.T, m.up_proj.weight.T], dim=-1)
    for m in per_expert_modules
])                                                    # [E, H, 2I]
down = torch.stack([m.down_proj.weight.T for m in per_expert_modules])  # [E, I, H]
```

若目标布局是 `[E, 2I, H]`（不转置直接 stack），拼装与 forward 都按第 3 节的配对规则相应调整——两者必须一起改。

改造清单：权重定义、拼装逻辑、forward 融合版本，三者一致；走 DCP 路径的模型，`checkpoint/fsdp/custom_model_converter/` 下的转换器一并调整。

改造完成后不要直接开 EP。先在 `expert_parallel_size=1` 下，用同种子对照**改造前后**两个版本的 loss——此时 EP 尚未参与，差异只能来自改造本身，从而把「改造引错」与「EP 配错」隔离成两个独立可定位的问题。通过后模块已是融合形态，回到阶段 1d 重新判定落入分支 A 还是 B（两闸门的完整定义见 SKILL.md 阶段 2a）。

---

## 3. 权重布局：配对规则与后果

仓内两种融合布局并存（wan2_2 为 `[E, H, 2I]` / `[E, I, H]`；minimax_m3_vl、gemma4 为 `[E, 2I, H]` / `[E, H, I]`）。**框架不做布局转换**——EP/EFSDP 分片对布局无感知，只按维序号切。因此布局是模型侧的自由度，识别它有两个直接用途：

**用途一：布局与 forward 是配对的。**

- `x @ w[e]` 要求每专家权重为 `[in, out]` → 布局 `[E, in, out]`；
- `F.linear(x, w[e])` 要求 `[out, in]` → 布局 `[E, out, in]`。

改布局不改 forward（或反之），轻则形状报错，重则维度恰好可广播而**悄然算错**。分支 C 改造、分支 B 写 `ep_forward` 时，布局、拼装、forward 三者必须按同一约定。

若 hidden 与 intermediate 相关维度数值相等，shape 无法唯一判定布局；此时以原生 forward 的 `F.linear` / `matmul` 语义为准，不使用 shape 自动推断方向。

**用途二：确定 EFSDP 的切分对象。** EFSDP 固定切第 1 维，布局决定被切的是 hidden 还是 intermediate。二者大小常差数倍，影响 `efsdp` 尺寸的整除约束与通信粒度。

**逐维判定示例**：config 给出 `num_experts=128`、`hidden_size=2048`、`intermediate_size=768`。

| 打印结果 | 逐维判定 | 结论 |
| --- | --- | --- |
| `gate_up_proj [128, 2048, 1536]`<br>`down_proj [128, 768, 2048]` | 128=专家数，2048=hidden，1536=2×768 | hidden 在第 1 维；forward 应为 `x @ w[e]` 风格 |
| `gate_up_proj [128, 1536, 2048]`<br>`down_proj [128, 2048, 768]` | 128=专家数，1536=2×768，2048=hidden | intermediate 在第 1 维；forward 应为 `F.linear` 风格 |

对不上时说明 config 字段读错（各模型命名不统一），回到阶段 1a 核对，不要假设。
