# EP 配置耦合与排障

阶段 3 配置、阶段 4 预检、阶段 5 排查时读本文。

EP 涉及 `parallel:` 段的七个字段：`expert_parallel_size`、`expert_fully_shard_parallel_size`、`ep_plan.apply_modules`、`ep_plan.apply_efsdp_modules`、`ep_plan.dispatcher`、`ep_plan.use_npu_fused_ops`、`fsdp_plan.apply_modules`，外加模型侧的专家总数与 top-k。它们之间有六条隐式耦合：改一个会牵动另一个，配置文件本身看不出来，报错位置也常远离根因。

## Source anchors

| 锚点 | 文件 |
| --- | --- |
| `ParallelApplier.apply_ep_modules`（剔除逻辑、自动填充） | `mindspeed_mm/fsdp/distributed/torch_parallelize.py` |
| `ParallelArguments.model_post_init`（尺寸推导） | `mindspeed_mm/fsdp/params/parallel_args.py` |
| `ParallelState.__post_init__`、`get_last_mesh_dim`（mesh 构建） | `mindspeed_mm/fsdp/distributed/parallel_state.py` |
| `dispatch_preprocess`、`ep_mc2_forward`（运行期校验） | `mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py` |

各字段当前默认值一律以代码为准，本文不写死。**使用某节前，先 grep 确认该节依赖的锚点仍存在**：锚点已改名或删除，说明框架在本文写成后已演进，该节结论可能失效——以当前代码为准重新核实，并提醒用户更新本文。锚点的作用是让文档过期变得可检测。

---

## 六条耦合

| # | 关联 | 规则 | 违反后果 |
| --- | --- | --- | --- |
| 1 | `expert_parallel_size` ↔ 专家总数 | 整除 | 运行期报错，但发生在训练启动后的 dispatch 阶段 |
| 2 | `expert_parallel_size` × `expert_fully_shard_parallel_size` × `edp` = world_size | `edp` 由 world_size 反推；`expert_fully_shard_parallel_size` 留空时的填充规则读 `model_post_init` | 并行状态初始化即报错 |
| 3 | `expert_parallel_size` ↔ `dispatcher` ↔ top-k | 默认 `alltoall`，除非用户指定。性能调优时按仓内经验（出处 `examples/qwen3_5/README.md`）：ep < top-k 时 `allgather` 更优——EP 组小，每卡本就要处理大部分 token，全量收集比按需交换划算；`mc2` 为昇腾通信计算融合路径，需显式指定 | **不报错，只慢**。这是性能选型而非硬约束 |
| 4 | `dispatcher: mc2` → `use_npu_fused_ops: true` | 强制 | 运行期报错，信息明确 |
| 5 | `ep_plan.apply_modules` ↔ `apply_efsdp_modules` ↔ `fsdp_plan.apply_modules` | 见下节 | 剔除失效 → 重复 `fully_shard`，报错在 FSDP 包装阶段，与 EP 配置无字面关联 |
| 6 | `apply_modules` 命中的模块 ↔ 权重布局 ↔ EFSDP `Shard(1)` | 布局与 forward 的配对规则见 `ep-module-contract.md` | **不报错，精度错** |

## 耦合 5 详解：三个 apply_modules 的分工

机制（读 `ParallelApplier.apply_ep_modules` 确认当前实现）：

1. `apply_efsdp_modules` 留空时自动填充为 `apply_modules`。
2. EP 与 EFSDP 施加后，框架从 `fsdp_plan.apply_modules` 中**剔除**已被 EFSDP 处理的条目，防止同一模块被 `fully_shard` 两次。
3. **剔除按配置中的模式字符串比较，不比较实际匹配到的模块。**

仓内惯例（qwen3_5 系列、qwen3_6、kimik2_5）：`fsdp_plan.apply_modules` 与 `ep_plan.apply_modules` 写**同一条字符串**。`fsdp_plan` 保留专家条目是为了维持 prefetch 所需的模块顺序（配置注释明确要求不随意调整该顺序），实际分片由剔除逻辑交给 EFSDP。

由此得出两条规则：

- **不要**为「职责分开」把专家条目从 `fsdp_plan` 删掉——会改变 prefetch 顺序。
- 两处都写时**逐字节相同**。语义等价但写法不同（空格、通配、层级前缀差异）会使剔除失效。

---

## 静默失效清单

以下情形**不报错、训练照常完成**，是排查的第一站：

| 情形 | 发现方式 |
| --- | --- |
| `expert_parallel_size > 1` 但 `apply_modules` 为空，EP 整段跳过 | 预检第 3 项；日志无 apply 记录；显存吞吐与 `ep=1` 无差异 |
| pattern 命中 MoE block，router 被误分片 | 打印命中模块名核对 |
| 布局与 forward/EFSDP 假设不符 | 阶段 5 对照超标；打印实际 shape |
| dispatcher 选型不当 | 只是慢，对比吞吐 |

会报错的情形（零命中、不整除、mesh 不匹配、dispatcher 非法、mc2 未开融合、重复 `fully_shard`）相对安全——有信号可循。

## 排障对照表

| 现象 | 先查 | 再查 |
| --- | --- | --- |
| 启动报 world size 不能被 mesh 组整除 | 耦合 2 乘积 | 是否显式配了 `expert_fully_shard_parallel_size` |
| 报 `No module named ...` | pattern 与实际 FQN | 通配写法（对照 `examples/` 已有配置） |
| 报专家数不能被 EP size 整除 | 耦合 1 | config 字段名是否读对 |
| 报 mc2 只支持融合模式 | 耦合 4 | — |
| FSDP 包装阶段报重复 `fully_shard` | 耦合 5 逐字节一致性 | 空格、通配、层级前缀的细微差异 |
| 跑通但显存吞吐与 `ep=1` 无差异 | 静默失效第 1 项 | 日志 apply 记录 |
| 跑通但阶段 5 对照超标 | 按 SKILL.md 阶段 5 排查顺序 | — |

## 报错翻译

EP 路径的报错常发生在 HCCL、DTensor 或 FSDP 包装层，与配置字段无字面关联。向用户汇报时先用上表映射回具体字段，说明哪个字段错了、应改成什么、为什么；不要原样转述底层堆栈。
