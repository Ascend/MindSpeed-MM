# MindSpeed-MM 代码优化 · 定位与核查

## 从仓库入口定位

根据 [仓库架构](../../../knowledge/architecture.md) 确认入口和消费者。以下路径用于定位影响范围，能否提出候选取决于改动是否满足 [验收标准](../SKILL.md#验收与产出)。

| 场景 | 排查入口 |
| --- | --- |
| Megatron | 从根目录 `pretrain_*.py`、`mindspeed_mm/training.py` 追踪到模型、数据 provider 和 forward。 |
| FSDP2 | 从 `mindspeed_mm/fsdp/train/trainer.py` 追踪 YAML 的 `training.plugin`、`ModelHub`、数据构建与并行配置。 |
| 共享链路 | 根据实际调用方确认影响范围。优先关注数据模板与 collator 的输出、checkpoint 键名及张量布局、通信的进程组与操作顺序。 |

## 扫描线索

| 候选 | 可选线索 |
| --- | --- |
| 重复定义、死代码、冗余 import 或赋值 | 找生效定义和真实引用，核对定义或表达式求值是否有副作用。 |
| 超长函数、深嵌套、重复逻辑 | 从主要调用链读控制流，确认异常、执行顺序以及局部状态的生命周期。 |
| 现有工具可替代的手写逻辑 | 对照同目录或公共模块的实现，核对参数、默认值和边界行为。 |

## 核查示例

- 同名定义被遮蔽时，核对实际生效的定义、默认参数与装饰器等定义期行为。示例报告的 `VideoDiTBlock.set_chunk_feed_forward` 是可证明等价的删除，`MOEModel.sharded_state_dict` 两份实现含不同的权重处理，值得作为疑似缺陷单独记录。
- FSDP2 的 `training.plugin` 会动态导入模块，装饰器在导入时注册模型和数据集。检查 import 或符号变更时，可沿 YAML、注册表或导出关系追踪真实消费者。
- 修改类的方法或属性时，可检查子类是否通过 `super()` 或属性访问复用实现。完整覆盖方法且不复用相关状态的子类，通常无需配套修改。
- 删除表面未用的赋值时，检查右侧调用是否有异常或副作用。示例报告将 `get_rank(process_group)` 列为跳过项，因为提前抛错也是可观察行为。
