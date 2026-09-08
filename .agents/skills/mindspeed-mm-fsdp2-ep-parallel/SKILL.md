---
name: mindspeed-mm-fsdp2-ep-parallel
description: Use when a MoE model on the MindSpeed-MM plugin-style FSDP2 backend needs expert parallelism — deciding ep_plan.apply_modules, choosing a dispatcher, reshaping per-expert weights into fused expert tensors, and proving EP is numerically correct rather than merely running. Use this even when the user does not say "EP" — reach for it whenever a MoE model needs to scale across devices, when expert weights won't fit, when someone asks why enabling expert_parallel_size changed nothing, or when a MoE model's loss diverges after a parallelism change. Triggers： "接入 EP", "开启专家并行", "配置 ep_plan", "expert parallel", "专家权重放不下", "EP 开了没效果", "EP 精度对不齐".
---

# MindSpeed-MM FSDP2 专家并行接入

把一个已在 FSDP2 后端跑通训练链路的 MoE 模型接入专家并行（EP）。

## 适用范围

EP 的施加逻辑位于 `mindspeed_mm/fsdp/distributed/`，属于框架核心，本技能通过配置驱动它，不修改它。若判断必须修改框架代码，**先停下并向用户说明**——这类需求通常意味着前面某步判断有误（pattern 指错模块、布局判定错误），而非框架缺陷。

只适用于 FSDP2 插件式后端，不涉及 Megatron/mcore 路线。

## 硬性规则（不可协商）

1. 不修改「代码修改范围」之外的任何文件。确需修改时升级给用户，不自行决定。
2. 完成判据是阶段 5 的等价性对照：bf16 下开关 EP 的 loss 平均相对差 ≤ 1%。「训练跑通」不构成判据，EP 配错通常不崩溃。
3. 对照两组必须共享同一逻辑初始权重、同种子、同数据顺序、同 world_size，且除 `expert_parallel_size` 外配置与环境一致；完整执行并报告数值。
4. 生效确认先于数值对照。EP 静默未生效时两组配置实际相同，对照会以「完美一致」假通过。
5. 分支 C 的权重布局改造与 EP 接入分两个闸门验证，不合并。
6. 除 EP balance（默认开启）外，首版不叠加 chunk loss、activation offload、LoRA 等特性；通过阶段 5 后逐个叠加并重跑对照。
7. 对照超标时排查根因，不放宽 1% 判据。
8. 最终验证必须在目标 `expert_parallel_size` 下进行。降级配置（`ep=1`、单步迭代、冻结层）只用于定位。
9. 预检项无法执行时标注「未验证」并说明原因，不得默认通过。
10. 无验证环境时，交付止步于配置方案与预检结论，明确说明缺何种条件；不得以「逻辑上应无问题」替代验证。

## 前置条件

1. 模型已完成 FSDP2 迁移，且跑通最小 E2E。未完成则先用 `mindspeed-mm-fsdp2-model-only-migration`。
2. 模型为 MoE 架构：存在 router/gate 与多个专家。
3. 具备多卡昇腾环境、可构建模型的配置与数据样本。真实权重非必需；无权重时按阶段 5 复用同一份初始化权重。

## 约定术语

| 术语 | 定义 |
| --- | --- |
| **专家容器** | 持有全部专家权重的模块，在模块树中是 router/gate 的同级兄弟节点 |
| **融合专家权重** | 单个 `nn.Parameter`，第 0 维索引专家，形如 `[E, ·, ·]`。EP 分片按第 0 维切，此形态是接入前提 |
| **per-expert 子模块** | `nn.ModuleList`，每个专家一组独立子模块。无法表达「按专家维切分」，需先改造 |
| **默认 EP forward** | 框架为未定义 `ep_forward` 的专家模块注入的前向实现（锚点 `get_experts_forward_fn_for_qwen`），按 Qwen 系 MoE 的结构假设编写。仓内 Qwen 系模型不改模型代码即可接 EP，原因就在这里 |
| **EP / EFSDP** | 专家维分片（切第 0 维，`ep` mesh）/ 专家内分片（切第 1 维，`efsdp` mesh），后者叠加在前者之上 |

## 并行底座

两张独立 mesh（`parallel_state.py`）：主 mesh（`dp_replicate × dp_shard × ulysses × ring × tp`）管非专家参数（`fsdp_plan`）；专家 mesh（`edp × efsdp × ep`）管专家参数（`ep_plan`），EP 切 `ep` 维、EFSDP 叠加切 `efsdp` 维。未配置的维由框架反推（`efsdp` 在 `model_post_init`，`edp` 在 `get_last_mesh_dim`），三维乘积恒等于 world_size。

施加顺序固定：quantization → TP → EP → FSDP → prefetch（`ParallelApplier.__call__`）。EP 仅当 `expert_parallel_size > 1` 且 `apply_modules` 非空才施加，否则静默跳过；EFSDP 条目随后按模式字符串从 `fsdp_plan` 剔除，防重复 `fully_shard`——阶段 3「逐字节相同」规则的根源。

## 工作流

```text
阶段 1 探测 -> 阶段 2 模型改造（仅 B/C）-> 阶段 3 配置 -> 阶段 4 预检
           -> 阶段 5 验证（生效确认 + 等价性对照，完成判据）
```

---

## 阶段 1：探测

在写任何配置之前完成，把后续所有决策依赖的事实一次性确定下来。

模型可用训练 YAML 走 `ModelHub` 构建；仅需模块树与参数形状时，直接从 `model_name_or_path` 构建并配合 meta device 即可，无需占用显存。

### 1a 配置探测

从模型 config 读取：专家总数、top-k、hidden size、intermediate size。

字段命名各模型不统一（`num_experts` / `num_local_experts` / `num_experts_per_tok` 等），按实际 config 读取，不套用其他模型的命名。

### 1b 模块树探测

`named_modules()` 定位专家容器：找到 router/gate，取其同级兄弟节点。

```text
model.language_model.layers.0.mlp                 <- MoE block（父节点，不选）
model.language_model.layers.0.mlp.gate            <- router
model.language_model.layers.0.mlp.experts         <- 专家容器（选此节点）
model.language_model.layers.1.mlp.experts
```

层号替换为通配，得到 pattern：`model.language_model.layers.{*}.mlp.experts`

选父节点的后果：EP 分片递归作用于被匹配模块下的全部参数，`gate` 权重会被按第 0 维切开。router 第 0 维通常正是专家数，切分后每卡只持有部分 router 输出——**不报错，精度错**。

### 1c 权重布局探测

`named_parameters()` 打印专家权重形状，用 1a 的四个数逐维对照，判定第 1 维是 hidden 还是 intermediate。

需要判定的原因：仓内两种布局并存，而框架不做任何布局转换——分片只按维序号切。布局决定 EFSDP（固定切第 1 维）切的是哪个量、`efsdp` 尺寸需整除哪个数，也必须与 forward 的矩阵乘约定配对。逐维判定示例与配对规则见 `references/ep-module-contract.md`。

对不上时说明 config 字段读错，回到 1a 核对，不要假设。

### 1d Forward 契约探测

框架接管专家模块前向有两条路径：模块自带 `ep_forward` 时优先调用它；否则注入默认 EP forward。默认实现按 Qwen 系 MoE 的结构编写，这是分支判定的基准。

读 `get_experts_forward_fn_for_qwen` 与本模型专家模块的 `forward`，对照三点：

1. **权重属性**：默认实现直接读取 Qwen 风格的融合权重属性与元数据属性（具体名称以函数体为准）。
2. **激活与门控**：dispatcher 默认激活为 SwiGLU（未传 `activation` 参数时）。本模型若用其他激活（如 GELU），即为偏离。
3. **调用约定**：MoE block 以「隐藏状态、路由权重、路由索引」三个位置参数调用专家容器。

三点全部一致 → 分支 A；任一偏离 → 分支 B。仅属性名不一致而其余一致时，优先在模块上补别名属性（代价最小），仍不满足再走 B。详细对照表见 `references/ep-module-contract.md`。

### 探测报告

固定结构输出，后续阶段直接引用：

```text
专家容器 FQN pattern : model.language_model.layers.{*}.mlp.experts
专家总数 / top-k      : 128 / 8
权重形态              : 融合专家权重 | per-expert 子模块
权重形状与布局        : gate_up_proj [128, 2048, 1536]，hidden 在第 1 维
Forward 契约          : 符合默认实现 | 偏离点：<具体说明>
已有 ep_forward       : 是 | 否
分支                  : A | B | C
```

### 分支判定

| 权重形态 | Forward 契约 | 分支 | 下一步 |
| --- | --- | --- | --- |
| 融合专家权重 | 三点全部一致 | A | 无需改模型，直接阶段 3 |
| 融合专家权重 | 任一偏离 | B | 阶段 2b |
| per-expert 子模块 | — | C | 阶段 2a |

---

## 阶段 2：模型改造

**分支 A 跳过本阶段**：默认 EP forward 已覆盖，直接进入阶段 3。

### 2a per-expert 子模块 → 融合专家权重（分支 C）

改造三处，具体步骤与 Qwen 风格代码示意见 `references/ep-module-contract.md`：

1. 权重定义：`ModuleList` 换为融合 `nn.Parameter`，布局对齐仓内同类模型，不引入新布局。
2. 权重加载：`from_pretrained` 按专家维拼装 per-expert 权重（注意 `nn.Linear.weight` 的 `[out, in]` 形状与目标布局的转置关系）；DCP 路径的转换器一并调整。
3. forward 同步改写为融合权重版本。

**改造与 EP 分两个闸门验证**（硬性规则 5），目的是隔离变量——若合并验证后对照失败，无法区分是改造引错还是 EP 配错：

- 闸门一（验证改造）：保持 `expert_parallel_size=1`，同种子对照**改造前后**两个版本各 10～20 步 loss，判据同阶段 5（平均相对差 ≤ 1%）。此时 EP 尚未参与，差异只能来自改造本身。
- 闸门二（验证 EP）：闸门一通过后，模块已是融合形态，回到阶段 1d 重新对照 forward 契约——改写的 forward 若按 Qwen 风格则落入分支 A，否则按分支 B 补 `ep_forward`；随后走阶段 3～5。

### 2b 实现 `ep_forward`（分支 B）

专家模块定义 `ep_forward` 后，框架优先调用它。签名与调用结构对齐仓内实现（见「参考实现」），偏离点通过 dispatcher 的既有参数表达（如自定义激活经 `activation` 参数传入，以 `ep_dispatcher.ep_forward` 当前签名为准），不重写 dispatcher。

在原模型类上做最小改动。不要用外层 wrapper 代理属性与 forward 重建模块树——那会破坏 state dict 与 checkpoint 语义。

实现后先用固定输入、路由结果和权重，对照原生 forward 与 `ep_forward` 的输出及梯度；局部契约未对齐时不要反复启动完整 E2E。

---

## 阶段 3：配置 `ep_plan`

位置：`examples/<model>/<model>_config.yaml` 的 `parallel:` 段（EP balance 开关在 `features` 段）。

| 字段 | 取值依据 |
| --- | --- |
| `expert_parallel_size` | 整除专家总数，且不大于卡数。性能调优推迟到阶段 5 通过之后 |
| `expert_fully_shard_parallel_size` | 留空由框架推导。显式配置前先算清 mesh 乘积 |
| `ep_plan.apply_modules` | 1b 得到的 pattern |
| `ep_plan.apply_efsdp_modules` | 留空，自动等于 `apply_modules` |
| `ep_plan.dispatcher` | **默认 `alltoall`，除非用户指定**。性能调优阶段可按仓内经验调整：`expert_parallel_size` 小于 top-k 时 `allgather` 通信量更优（出处 `examples/qwen3_5/README.md`）；`mc2` 为昇腾通信计算融合路径，需显式指定 |
| `ep_plan.use_npu_fused_ops` | 选 `mc2` 时必须为 true |
| `fsdp_plan.apply_modules` | 仓内惯例保留专家条目以维持 prefetch 顺序；此时该条目必须与 `ep_plan.apply_modules` **逐字节相同**，机制见 `references/ep-config-coupling.md` 耦合 5 |
| `enable_ep_balance` | **默认 `true`，除非用户指定关闭**（`features` 段，字段定义以 `feature_args.py` 为准） |

七个字段之间存在六条隐式耦合——改一个会牵动另一个，而配置文件本身看不出来。写完配置后对照 `references/ep-config-coupling.md` 核查。

正式 EP balance 只通过 `features.enable_ep_balance` 与 `ep_balance_plan` 配置。`MM_FORCE_EP_BALANCE=1` 会直接覆写路由索引，仅用于调试强制均衡；正式对照必须保持未设置。

除 EP balance 外不叠加其他特性（硬性规则 6）。

---

## 阶段 4：预检

| # | 检查项 | 违反后果 |
| --- | --- | --- |
| 1 | 专家总数能被 `expert_parallel_size` 整除 | 运行期报错，发生在 dispatch 阶段 |
| 2 | 专家 mesh 三维乘积等于 world_size（`edp × efsdp × ep`；推导规则以 `ParallelArguments.model_post_init` 为准） | 并行状态初始化即报错 |
| 3 | `ep_plan.apply_modules` 非空 | **静默跳过全部 EP，训练照常完成。头号失败模式** |
| 4 | `dispatcher` 取值合法；`mc2` 时 `use_npu_fused_ops` 为 true | 运行期报错，信息清晰 |
| 5 | 每条 `apply_modules` 至少命中 1 个模块，且命中专家容器而非 MoE block 或 shared expert | 零命中报错；命中错模块**不报错，精度错** |
| 6 | 两处专家条目字符串逐字节相同 | 重复 `fully_shard`，报错位置远离根因 |

第 5 项需构建模型；其余读配置与模型 config 即可。

---

## 阶段 5：验证（完成判据）

两组训练运行覆盖全部验证目标——基准组与待验组本身就是两次完整 E2E，生效确认与等价性对照都在其中完成。

固定种子与数据顺序，同一 world_size 下各跑 10～20 步，其余配置完全一致：

| 组 | 配置 |
| --- | --- |
| 基准 | `expert_parallel_size = 1` |
| 待验 | `expert_parallel_size = N`（目标值） |

两组都应满足：loss 有限且非 NaN / Inf / 恒零。

两组必须从同一逻辑权重开始：有 checkpoint 时加载同一份；无权重时只初始化一次并让两组复用该状态。不要在不同 mesh 下分别随机初始化后直接对照，同 seed 不保证逻辑权重相同。

### 5a 生效确认（先于对照，硬性规则 4）

顺序不可颠倒的原因：若 EP 静默未生效（预检第 3 项的失败模式），待验组实际就是 `ep=1`，两组 loss 会「完美一致」——对照假通过。因此先拿出生效证据，再看数值：

- 日志或运行期检查列出实际命中的专家模块名与数量，并确认使用 EP forward。
- 专家 mesh 各维尺寸正确，专家权重的 DTensor placement / local shape 符合 EP 分片预期。
- 记录显存与吞吐相对基准组的变化，作为辅助证据；异常时回到预检第 3、5 项。

### 5b 等价性对照（判据，bf16）

EP 在数学上是不改变语义的纯并行变换：相同输入与种子下，开关 EP 应得到相同的训练轨迹。切错维、布局配对错误、dispatcher 路由错这三类典型错误均不崩溃，只有依此不变量对照才能暴露。

**逐步计算 `|待验 − 基准| / |基准|`，全程平均 ≤ 1% 为通过。**

首步应几乎一致。解释首步差异前先确认初始权重与控制变量一致；确认后，首步的明显差异才指向前向路径，无需跑满即可停下排查。超标即不通过，进入排查；不放宽判据（硬性规则 7）。

### 排查顺序

靠前的原因会掩盖靠后的现象。先做低成本检查与局部对照，每次只改变一个变量，定位后再重跑完整 E2E：

1. EP 是否真的生效（5a）。
2. 两组初始权重、配置、环境与逐步学习率是否一致。
3. 权重布局与 forward 是否配对、EFSDP 切维是否符合预期（`references/ep-module-contract.md`）。
4. 匹配到的是专家容器还是 MoE block。
5. dispatcher：换一种取值看是否复现。若换后正常，问题在 dispatcher 路径而非分片。
6. 叠加特性隔离：关闭 EP balance、router aux loss 后复跑对照，排除其实现的干扰。
7. 梯度与通信：梯度除法因子的自动填充值；HCCL 不支持 `ReduceOp.PREMUL_SUM`，EFSDP 路径将其 patch 为 SUM 加标量乘（锚点 `apply_hccl_premul_sum_patch`）。定位依据：初始权重一致时，首步 loss 计算于任何权重更新之前，梯度路径的错误不可能影响首步，只能从第二步起通过污染权重显现——**首步即偏查前向（第 3～5 项），首步一致而差值随步数放大查本项**。

ST 基线与性能验证不在本技能范围；具备昇腾 CI 环境时建议补做，入口与门限见 `tests/README.md`。

---

## 参考实现

### 分支 A：零模型改动

两条路径都落入分支 A，配置写法相同：模型自带 `ep_forward`（框架优先绑定），或 forward 契约三点全部符合默认实现。

- 自带 `ep_forward` 的代码参照：`mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py`（其 `ep_forward` 含 EP balance 接入）。注意它属于自带路径——不要拿它做 1d 的三点对照基准，基准是 `get_experts_forward_fn_for_qwen`。
- 走默认 forward 的配置参照：`examples/kimik2_5/kimik2_5_config.yaml`：

```yaml
parallel:
  fully_shard_parallel_size: auto
  fsdp_plan:
    apply_modules:
      # ...（vision_tower 等其余条目见原文件，顺序勿随意调整）
      - language_model.model.layers.{*}
      - language_model.model.layers.{*}.mlp.experts   # 与 ep_plan 逐字节相同
  expert_parallel_size: 16
  ep_plan:
    apply_modules:
      - language_model.model.layers.{*}.mlp.experts
```

要点：模型代码零改动（分支 A 的意义所在）；`apply_efsdp_modules` 留空；专家条目在两处以相同字符串出现；`dispatcher` 未显式配置时取默认值 `alltoall`。该配置同时示范 shared expert 的处理：pattern 只指 `mlp.experts`，同级 `mlp.shared_experts` 不被匹配。其余模型的现成配置在 `examples/` 下 grep `ep_plan` 检索。

### 分支 B：wan2_2

`mindspeed_mm/fsdp/models/wan2_2/modeling_wan2_2.py` 的 `WanMoEExperts`，是自定义 `ep_forward` 的标准参考之一。

走分支 B 的原因：专家激活为 GELU(tanh)，偏离默认的 SwiGLU。处理方式是把自定义激活经 dispatcher 的 `activation` 参数传入（参数名以 `ep_dispatcher.ep_forward` 当前签名为准），而非重写 dispatcher；同时补 `hidden_size` 别名属性满足元数据要求。其结构（取本地分片 → 定义偏离点 → 调用 dispatcher → 还原形状）是分支 B 的标准样板。

---

## 代码修改范围

按分支渐进，后一层包含前一层：

| 层 | 分支 | 路径 | 内容 |
| --- | --- | --- | --- |
| **0** | A / B / C | `examples/<model>/<model>_config.yaml` | EP 相关配置；验证所需的其他字段仅做最小调整并在报告说明 |
| **1** | B / C | `mindspeed_mm/fsdp/models/<model>/modeling_*.py` | `ep_forward` 及其必要的属性、配置传播与 EP balance 接入 |
| **2** | C | 同上 | 专家权重定义改为融合张量；`from_pretrained` 拼装逻辑 |
| **2** | C | `checkpoint/fsdp/custom_model_converter/<model>.py` | 权重转换器（仅当该模型走 DCP 路径且已有转换器时） |

**升级区**：以上之外的一切，尤其是 `mindspeed_mm/fsdp/distributed/**`、`trainer.py`、`train_engine.py`、`ModelHub`、通用 dataloader 构建器。修改前停下并向用户说明必要性。

验证完成后生成 `examples/<model>/<model>_ep_validation_report.md`；该报告是验证产物，不属于实现代码修改范围。

## 按需参考

- `references/ep-module-contract.md` —— 阶段 1c/1d 判定布局与契约、阶段 2 改造模块时读。含分支 C 改造代码示意。
- `references/ep-config-coupling.md` —— 阶段 3 配置、阶段 4 预检、阶段 5 排查时读。含六条耦合与排障对照表。

## 反模式

| 问题 | 修正方式 |
| --- | --- |
| 设了 `expert_parallel_size` 但 `ep_plan.apply_modules` 为空 | EP 静默跳过。预检第 3 项 + 5a 生效确认 |
| 以「训练跑通」作为 EP 生效或正确的证据 | 跑通只证明链路完整。需 apply 记录、mesh 尺寸、显存吞吐差异三项证据 |
| 未做生效确认就报告对照通过 | EP 静默未生效时对照必然「通过」，是假阴性。5a 先于 5b |
| `apply_modules` 指向 MoE block | router 被按专家维切开且不报错。打印命中模块名核对 |
| pattern 误含 shared expert（`shared_expert*` 兄弟节点） | 稠密权重被按第 0 维（hidden 维）切开，不报错、精度错。按 1b 用专家维消歧兄弟节点 |
| 模型已有 `ep_forward` 却按分支 B 另写一套 | 框架优先绑定已有 `ep_forward`。1d 先查存在性，已有即分支 A |
| 套用其他模型的权重布局或 config 字段名 | 仓内均不统一。逐模型执行阶段 1 探测 |
| 两处专家条目字符串不一致 | 重复 `fully_shard`。逐字节比对 |
| 对照两组的初始权重、种子、数据顺序、world_size 或其他配置不一致 | 差异无法归因于 EP，对照作废。除 `expert_parallel_size` 外必须相同 |
| 正式对照设置 `MM_FORCE_EP_BALANCE=1` | 它会覆写路由索引，不是正式 balance 开关；取消该环境变量 |
| 对照超标即调整判据或改口宣称通过 | 1% 是判据不是建议。按阶段 5 排查顺序定位根因 |
| 以降级配置跑通后宣称完成 | 降级只用于定位。最终验证须在目标 `expert_parallel_size` 下 |
| 分支 C 将布局改造与 EP 合并验证 | 拆为两个闸门（2a） |
| 分支 B 重写 dispatcher 以适配偏离点 | 通过 dispatcher 既有参数（如 `activation`）传入，参见 wan2_2 |
| 原样转述 HCCL / DTensor / FSDP 层报错 | 用 `references/ep-config-coupling.md` 排障表映射回具体配置字段 |
| 为接入 EP 修改框架核心 | 通常意味着阶段 1 判断有误。先复核，仍需修改则升级给用户 |

## 输出模板

将以下内容写入验证报告，并向用户提供结论摘要：

````markdown
## 探测报告
专家容器 FQN pattern : <...>
专家总数 / top-k      : <...>
权重形态              : <...>
权重形状与布局        : <...>
Forward 契约          : <...>
已有 ep_forward       : <是 | 否>
分支                  : A | B | C

## 配置
```yaml
<最终 parallel: 段与 features 段相关项>
```
dispatcher：<默认 alltoall | 其他取值及依据>；EP balance：<开 | 用户指定关>

## 预检
| # | 检查项 | 结论 |
（六项逐项给出；未验证项写明原因）

## 验证
- 初始化来源：<同一 checkpoint | 复用的同一初始化状态>
- 控制变量：<除 expert_parallel_size 外配置与环境一致；MM_FORCE_EP_BALANCE 未设置>
- 生效证据：<实际模块名与数量 / EP forward / mesh / DTensor local shape / 显存吞吐>

| step | 基准 loss | EP loss | 相对差 |
| --- | ---: | ---: | ---: |
| ... | ... | ... | ... |

- 等价性对照：基准与待验各 <N> 步，loss 平均相对差 <x>%
- 判定：通过 | 不通过（不通过时给出排查至第几步、下一步查什么）

## 修改范围
- 改动文件列表及所属层级（0 / 1 / 2）
- 是否触及升级区
- 已执行与未执行的验证项及原因
````
