# DDP

## 背景与挑战

FSDP2 沿 0 维切分参数（参见 [FSDP2](fsdp2.md)），显存占用最低，但分片本身有代价：

- 前向需要 all-gather、反向需要 reduce-scatter，小模型或小规模集群下通信占比高；
- 参数以分片 DTensor 形式存在，`state_dict` 与单卡模型不同，权重加载与转换链路更复杂；
- 单机验证、小模型调试或减层联调时，FSDP2 的分片收益有限，反而增加调参成本。

## 解决方案

MindSpeed MM 在 FSDP2 训练后端中内置 DDP 路线：当 `parallel.fully_shard_parallel_size` 为 1 时，框架不再对非专家模块执行 `fully_shard`，而是用 PyTorch 的可组合 DDP（`torch.distributed._composable.replicate`）复制参数：

- 不改变参数 dtype、不替换 forward，模型对外 API 与单卡一致；
- 不引入 FlatParameter，`state_dict` 的 key 与单卡模型完全一致，权重保存/加载沿用 FSDP2 的 DCP/HF 流程；
- MoE 模型可与专家并行组合：专家参数由 `ep_plan` 按 EP 切分，需要时再通过 `expert_fully_shard_parallel_size` 在 EP 组内做 FSDP 分片（EFSDP），非专家参数由 DDP 复制；嵌套的专家 FSDP 单元不进入 DDP reducer，其梯度由 FSDP2/EFSDP 负责同步；
- 梯度规约延迟到梯度累积的最后一个小微批再 all-reduce，避免每个小微批都做通信；嵌套的专家 FSDP 单元通过 `set_is_last_backward` / `set_requires_all_reduce` 保持同步；
- 数据并行组大小为 1（单卡）时不构建 reducer。

## 使用方法

DDP 与 FSDP2 共用训练入口 `mindspeed_mm/fsdp/train/trainer.py` 和同一套六段式 YAML 配置，不需要额外的命令行开关，只需把 `parallel.fully_shard_parallel_size` 置为 1：

```yaml
parallel:
  # 置为 1 表示非专家参数不做 FSDP 分片，走 DDP 复制
  fully_shard_parallel_size: 1

  fsdp_plan:
    # DDP 下保持为空：非专家参数整体交给 DDP
    apply_modules: []
    # DDP 不支持参数、梯度与优化器状态的 CPU 卸载
    cpu_offload: false
```

除 `parallel` 外的其余配置段（`model`、`data`、`features`、`training`、`tools`）与 FSDP2 完全一致，DDP 不引入额外配置项，按实际任务填写模型、数据集与训练超参即可，各字段含义见 [FSDP2](fsdp2.md)。启动方式也相同：`export NON_MEGATRON=true` 后用 torchrun 拉起 `mindspeed_mm/fsdp/train/trainer.py`。

### MoE 模型：与专家并行组合

MoE 模型可以在 DDP 的基础上让专家参数走 EP，非专家参数仍由 DDP 复制。在上述 `parallel` 段中追加：

```yaml
parallel:
  fsdp_plan:
    # 专家 FSDP 单元做重计算时需要，按模型实际情况填写
    hook_modules:
      - model.layers.{*}
    # 可选：专家侧混精，只作用于专家 FSDP 单元
    param_dtype: bf16
    reduce_dtype: fp32

  # 按实际部署填写，三者乘积需整除 world size
  expert_parallel_size: 8
  expert_fully_shard_parallel_size: 1
  expert_data_parallel_size: 1

  ep_plan:
    # 按模型实际的专家模块路径填写
    apply_modules:
      - model.layers.{*}.mlp.experts
    apply_efsdp_modules:
      - model.layers.{*}.mlp.experts
    dispatcher: alltoall
    use_npu_fused_ops: true
```

专家模块只需在 `ep_plan.apply_modules` 中声明，不要写进 `fsdp_plan.apply_modules`。

### 配置项说明

| 配置项 | 说明 |
| --- | --- |
| `parallel.fully_shard_parallel_size` | 置为 `1` 时启用 DDP；大于 `1` 时走 FSDP2 参数分片 |
| `parallel.fsdp_plan.apply_modules` | 需保持为空列表，非专家参数不参与 FSDP 分片；专家分片由 `ep_plan.apply_efsdp_modules` 单独指定 |
| `parallel.fsdp_plan.hook_modules` | 专家 FSDP 单元重计算所需的 hook 模块；非 MoE 模型无需配置 |
| `parallel.fsdp_plan.param_dtype` / `reduce_dtype` | 仅作用于专家 FSDP 单元；DDP 参数保持 `model.dtype`，不创建 FP32 master 参数 |
| `parallel.fsdp_plan.cpu_offload` | 必须为 `false`，DDP 不支持参数、梯度与优化器状态的 CPU 卸载 |
| `parallel.expert_parallel_size` 与 `ep_plan` | 仅 MoE 模型使用：专家参数走 EP（可选 EFSDP），与 DDP 组合生效 |

### 与 FSDP2 的差异

| 维度 | DDP | FSDP2 |
| --- | --- | --- |
| 参数分片 | 非专家参数不分片；专家参数按 EP 切分 | 按 `fsdp_plan` 沿 0 维分片为 DTensor |
| 显存占用 | 较高（非专家参数、梯度、优化器状态完整保留） | 低（按分片组均摊） |
| 通信 | 反向 all-reduce，梯度累积时每轮一次 | 前向 all-gather、反向 reduce-scatter |
| `state_dict` | 与单卡模型一致 | 分片 DTensor |
| 适用场景 | 小模型、减层调试、单机验证 | 大模型正式训练 |

## 使用效果

在小模型或减层调试场景下，DDP 省去了 FSDP2 的 all-gather / reduce-scatter，通信开销和配置复杂度更低，权重也与单卡模型直接对齐，便于快速验证收敛。由于非专家参数不分片，单卡可承载的模型规模明显小于 FSDP2。

## 注意事项

1. **DDP 不切分非专家参数，模型容易 OOM。** 每张卡都要容纳完整的非专家参数、梯度与优化器状态，显存占用显著高于 FSDP2。模型规模超出单卡容量时训练会 OOM，此时需要减层：将模型配置（如 HuggingFace `config.json` 中的 `num_hidden_layers`）调小后再初始化/加载权重；同时可开启重计算（`features.recompute`）、激活值卸载（`features.enable_activation_offload`），并降低 `training.micro_batch_size`、`data.dataset_param.basic_parameters.cutoff_len` 等，进一步压缩显存。减层后仍无法满足显存要求时，请改回 FSDP2（`fully_shard_parallel_size > 1`）让参数参与分片。
2. 启用 DDP 后 `fsdp_plan.apply_modules` 不再生效，请保持为空；不要将 FSDP2 的分片单元与 DDP 混用。
3. DDP 不支持 `fsdp_plan.cpu_offload: true`，配置后会直接报错。
4. DDP 不支持词表并行（`vocab_parallel_size > 1`）。
5. `param_dtype`、`reduce_dtype` 等混精配置只作用于专家 FSDP 单元；DDP 参数的精度由 `model.dtype` 决定，框架不会为其创建 FP32 master 参数。
6. DDP 的权重保存/加载沿用 FSDP2 的 DCP/HF 流程，`state_dict` key 与单卡模型一致，模型代码无需为 DDP 做额外适配。
7. MoE 模型可与 EP 组合：非专家参数由 DDP 复制，专家参数由 `ep_plan` 按 EP 切分（`expert_fully_shard_parallel_size > 1` 时再在 EP 组内做 FSDP 分片）。此时 `expert_parallel_size × expert_fully_shard_parallel_size × expert_data_parallel_size` 需整除 world size，专家模块只在 `ep_plan.apply_modules` 中声明。
