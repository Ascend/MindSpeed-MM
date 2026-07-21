# MindSpeed MM FAQ

| 序号 | 问题简述 | 影响 | 当前方案 | 后续计划 | 相关链接 |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 1 | CANN 版本不匹配导致训练报错 | 训练无法启动或出现算子报错 | 确保驱动固件、CANN Toolkit/Kernels/NNAL、PyTorch、TorchNPU版本严格配套，参考 [版本配套说明](./release_notes_mm.md#相关产品版本配套说明) | 持续更新版本配套表 | [安装说明](./pytorch/install_guide.md) |
| 2 | `ModuleNotFoundError: No module named 'acl'` | 无法导入昇腾相关模块 | 检查是否已执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，建议写入 `~/.bashrc` | - | [安装说明](./pytorch/install_guide.md) |
| 3 | CANN nnal 包安装顺序错误导致 `libatb.so` 找不到 | 推理或训练时报动态库缺失错误 | nnal 包必须在 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 之后安装，否则找不到依赖路径 | - | [安装说明](./pytorch/install_guide.md) |
| 4 | 多机多卡启动时脚本卡死无报错 | 无法启动分布式训练 | 确保已安装 `pdsh`，否则多机多卡启动脚本会卡死。可通过 `apt install pdsh` 或 `yum install pdsh` 安装 | - | - |
| 5 | HuggingFace 权重无法直接用于训练 | 训练启动失败 | HuggingFace 权重格式（safetensors/bin）不能直接用于 Megatron 架构训练，需使用 `mm-convert` 工具转换为 MindSpeed-MM 格式 | - | [权重转换](./features/mm_convert.md) |
| 6 | 权重转换时 TP/PP 参数与训练不一致 | 训练加载权重失败 | 权重转换时的 `tp_size`、`pp_size` 及 `llm_pp_layers`/`vit_pp_layers` 必须与训练脚本中的并行配置保持一致 | - | [权重转换](./features/mm_convert.md) |
| 7 | `tp_size` 超过 `num_key_value_heads` 导致报错 | 张量并行切分失败 | TP 并行度（`tp_size`）不能超过模型配置中的 `num_key_value_heads`，否则 KV 头无法均分到各卡 | - | - |
| 8 | 训练完成后权重文件数量与原始 HF 模型不一致 | 用户疑惑权重是否完整 | 训练生成的是 Megatron-Core (mcore) 格式权重，经 mg2hf 转换后得到标准 HF 格式。文件数量因切分策略不同而变化，但参数完整，需重新生成 `model.safetensors.index.json` | - | [权重转换](./features/mm_convert.md) |
| 9 | NPU 显存不足（OOM） | 训练中断 | 可采取以下措施：1. 减小 `micro-batch-size`（最低至1）；2. 增大 TP/PP 并行度（TP×PP≤NPU 数量）；3. 减小 `seq-length`；4. 开启重计算（`--recompute-granularity full --recompute-method block --recompute-num-layers`）；5. 使用 ChunkLoss 降低显存峰值 | - | [ChunkLoss](./features/chunkloss.md) |
| 10 | LLM PP 切分为 0 层时出现 `learning_rate=None` assertion 报错 | 训练启动后立即崩溃 | PP 切分配置中 LLM 部分不能出现 0 层的 stage，例如 `llm=[0,8,10,10]` 会导致该问题，需调整为 `llm=[1,7,10,10]` 等确保每个 stage 均有 LLM 层 | 优化学习率加载逻辑以支持 0 层 stage | - |
| 11 | 数据预处理超时或同步报错 | 大数据集训练无法启动 | 可增大超时参数 `--distributed-timeout-minutes`；对于超大数据集（百万级以上），建议分批预处理或使用更高性能存储 | 优化数据预处理并行效率 | - |
| 12 | 网卡名称错误导致通信超时 | 多机训练无法启动 | 使用 `ifconfig` 检查网卡名称，设置对应环境变量：`export HCCL_SOCKET_IFNAME=<网卡名>`、`export TP_SOCKET_IFNAME=<网卡名>`、`export GLOO_SOCKET_IFNAME=<网卡名>` | - | - |
| 13 | 保存 checkpoint 时超时报错 | 训练完成后保存权重失败 | 确保磁盘 IO 带宽正常，单个节点最大约60GB文件需在36分钟内保存完成；也可忽略该报错，不影响已保存的权重 | - | - |
| 14 | 复制脚本后出现 `syntax error near unexpected token` | 脚本无法执行 | 从 Windows 复制脚本到 Linux 时换行符不匹配，执行 `dos2unix xxx.sh` 或在 vim 中设置 `:set ff=unix` | - | - |
| 15 | 不同 CANN 版本环境变量冲突 | 多版本环境下训练异常 | 不同版本的 CANN 包建议使用 Docker 隔离，避免环境变量互相干扰 | - | - |
| 16 | 多模态模型训练中快慢卡负载不均衡 | 训练效率低下 | 使用多模态异构 PP 切分，将视觉编码器、音频编码器和 LLM 分别配置不同的 PP 层数分布，避免某些卡负载过重 | - | [异构并行](./features/hetero_parallel.md) |
| 17 | 训练 loss 不收敛 | 模型效果差 | 检查：1. 数据质量和清洗逻辑；2. 学习率和 warmup 策略；3. 是否使用预训练权重初始化；4. 数据路径和图片路径是否正确 | - | - |
| 18 | 推荐的关键环境变量配置 | 影响训练性能和稳定性 | 建议配置：`export ASCEND_GLOBAL_LOG_LEVEL=1`、`export TASK_QUEUE_ENABLE=2`、`export CPU_AFFINITY_CONF=2`、`export HCCL_CONNECT_TIMEOUT=600`、`export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | - | - |
| 19 | Megatron-LM 版本与 MindSpeed-MM 不匹配 | 安装或运行报错 | MindSpeed-MM 需要配套版本的 Megatron-LM，请参考 [安装说明](./pytorch/install_guide.md) 中指定的 commit 或 tag 进行 checkout | - | [安装说明](./pytorch/install_guide.md) |
| 20 | Docker 镜像中缺少模型特定依赖 | 模型训练报错 | Docker 镜像仅预装 torch、TorchNPU 和 decord 基础依赖，需根据目标模型的 README 在 base 环境中手动安装额外依赖 | - | - |
