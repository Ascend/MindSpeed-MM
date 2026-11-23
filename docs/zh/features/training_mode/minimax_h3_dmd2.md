# MiniMax-H3 DMD2 LoRA 训练与续训

MiniMax-H3 DMD2 在 MindSpeed-MM FSDP2 后端实现音视频分布匹配训练，支持新建 Student LoRA、导入已有 LoRA 续训，以及完整训练状态恢复。算法参考 [DMD2 论文](https://arxiv.org/abs/2405.14867)。

## 训练过程

Student 从噪声生成少步音视频，Teacher 和 Critic 在同一加噪样本上预测去噪结果。Student 按两者的分布差异更新；Critic 回归 Student 生成样本对应的速度目标。Teacher 固定，Student 与 Critic 使用独立优化器、学习率和更新频率。

视频与音频各自按整条样本计算归一化因子，再合并损失。双声道音频是一条联合样本，打包为 `[1, 2T, 32]` 或还原为 `[2, 32, T]` 时保持相同的归一化范围。归一化和伪目标使用 FP32，对极小分母设置数值下限。

训练复用 FSDP2 分片、重计算、LoRA 权重转换和 DCP 检查点。预计算缓存中的文本、首帧图片及位置布局传入 Student、Teacher 和 Critic。

## 配置

| 模式 | 配置文件 |
| --- | --- |
| 新建 LoRA | `examples/minimax_h3/posttrain_dmd2.yaml` |
| 已有 LoRA 续训 | `examples/minimax_h3/posttrain_dmd2_continue.yaml` |

填写以下路径：

| 配置项 | 用途 |
| --- | --- |
| `model.model_name_or_path` | 本地 MiniMax-H3 Transformer 配置与权重 |
| `data.dataset_param.basic_parameters.dataset_dir` | 预计算 `.pth` 缓存 |
| `distillation.student.import_lora` | 新建设为 `null`，续训填写源 LoRA 路径及缩放 |
| `training.save` | 检查点输出目录，多机使用共享目录 |

缓存格式为 `(shared, positive, negative)`，包含音视频 latent、文本 embedding 与模态位置布局。T2VA 使用文本条件；图生音视频对应 FL2VA，本示例仅使用首帧条件，通过 `keyframe_cond_anchor` 提供首帧图片。画幅和时长由缓存决定。依赖与数据处理见 [MiniMax-H3 示例](../../../../examples/minimax_h3/README.md)。

[新建 LoRA 配置](../../../../examples/minimax_h3/posttrain_dmd2.yaml)和[已有 LoRA 续训配置](../../../../examples/minimax_h3/posttrain_dmd2_continue.yaml)都使用八步 SDE 采样（`student_steps: 8`、`rollout_mode: sde`）。八步指 Student 生成轨迹的 8 个降噪时间点。训练时随机选一个时间点作为本次生成的终点；从一个时间点推进到下一个时，先预测干净的音视频潜变量，再按下一时间点的噪声强度注入新的随机噪声。训练迭代数由 `training.train_iters` 单独控制。

两份配置均使用视频/音频 shift=6/3、1 次 Student 对应 4 次 Critic 更新、速度回归目标。学习率分别为 1e-4/2e-4，LoRA rank/alpha 分别为 64/64、128/128，可在配置中调整。6 次 smoke 更新覆盖两次 Student 和四次 Critic 更新。

已有 LoRA 支持 LightX2V、DiffSynth 和 PEFT 格式，导入保留实际模块、rank 和缩放。LightX2V 的独立 Q/K/V 与 gate/up 权重转换到框架的融合布局，保留低秩增量。

```yaml
distillation:
  student:
    import_lora:
      path: ./checkpoints/student_lora.safetensors
      source_format: lightx2v
      alpha: 8
      scaling: alpha_over_rank

training:
  load: ""
  train_iters: 6
  save: ./outputs/minimax_h3/dmd2_smoke
  save_interval: 3
```

`alpha`、时间表和 shift 按所用源权重填写；更新次数按 Student/Critic 优化器更新总数计数。

## 启动

**启动前请检查并行配置：**当前 DMD2 仅支持 FSDP2 数据并行。请将 `tensor_parallel_size`、`ulysses_parallel_size`、`ring_attention_size`、`expert_parallel_size` 和 `kvallgather_parallel_size` 均设为 `1`。

在仓库根目录通过 `CONFIG_PATH` 选择配置启动：

```bash
CONFIG_PATH=examples/minimax_h3/posttrain_dmd2.yaml \
bash examples/minimax_h3/posttrain_minimax_h3_dmd2.sh
```

续训时将配置换成 `posttrain_dmd2_continue.yaml`。多机在每个节点执行一次，设置相同主节点地址和各自的节点编号：

```bash
CONFIG_PATH=examples/minimax_h3/posttrain_dmd2_continue.yaml \
NNODES=2 NODE_RANK=0 MASTER_ADDR="主节点IP" MASTER_PORT=23456 \
bash examples/minimax_h3/posttrain_minimax_h3_dmd2.sh
```

第二个节点设置 `NODE_RANK=1`，并将两节点的 `MASTER_ADDR` 都替换为实际主节点 IP。默认每节点 8 个 NPU 进程，可用 `NPUS_PER_NODE` 覆盖。

## 保存与恢复

检查点写入 `training.save/update_XXXXXXXX/`，包含 Student/Critic LoRA、优化器、学习率、随机状态和数据位置；`COMPLETED` 标识完整保存。恢复同一任务时设置 `distillation.resume_from` 为检查点目录。启动日志以 `[DMD2]` 标识角色、损失和更新次数，并通过 `init=fresh_lora` / `init=imported_lora` 显示初始化方式。
