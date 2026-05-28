# 故障排查指南

本文档按场景分类提供 MindSpeed MM 常见问题的排查方法，帮助用户快速定位和解决问题。

## 安装故障排查

### 环境变量未生效

**现象**：执行训练脚本时报 `ModuleNotFoundError: No module named 'acl'` 或类似错误。

**排查步骤**：

1. 检查是否已执行 CANN 环境变量初始化：

   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

2. 建议将上述命令写入 `~/.bashrc`，避免每次手动执行
3. 验证环境变量是否生效：

   ```bash
   echo $ASCEND_HOME_PATH
   echo $LD_LIBRARY_PATH
   ```

### CANN 版本不匹配

**现象**：训练启动报错或出现算子错误。

**排查步骤**：

1. 检查当前 CANN 版本：

   ```bash
   cat /usr/local/Ascend/cann/version.info
   ```

2. 对照 [版本配套说明](release_notes_mm.md#相关产品版本配套说明) 确认版本是否配套
3. 检查 torch_npu 版本是否与 CANN 版本匹配：

   ```bash
   python -c "import torch_npu; print(torch_npu.__version__)"
   ```

4. 如版本不匹配，按配套表重新安装对应版本

### nnal 包安装后 libatb.so 找不到

**现象**：`OSError: libatb.so: cannot open shared object file`

**排查步骤**：

1. 确认安装顺序：必须先 `source /usr/local/Ascend/cann/set_env.sh`，再安装 nnal 包
2. 安装 nnal 后再次 source 环境变量：

   ```bash
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

3. 验证库文件是否存在：

   ```bash
   find /usr/local/Ascend -name "libatb.so"
   ```

### Megatron-LM 版本不匹配

**现象**：`ImportError`、`AttributeError` 或运行时 API 不兼容。

**排查步骤**：

1. 检查 Megatron-LM 版本：

   ```bash
   cd Megatron-LM && git log --oneline -1
   ```

2. 确认是否 checkout 到正确版本（当前推荐 `core_v0.12.1`）
3. 检查 megatron 目录是否正确拷贝到 MindSpeed-MM 根目录

---

## 训练故障排查

### 训练 loss 不收敛

**现象**：训练 loss 不下降或持续震荡。

**排查步骤**：

1. **数据检查**：
   - 确认数据路径和图片路径是否正确
   - 检查数据质量和清洗逻辑
   - 验证数据预处理脚本是否正确执行
2. **超参检查**：
   - 学习率是否设置合理
   - warmup 策略是否正确配置
   - batch size 是否过大或过小
3. **权重检查**：
   - 是否使用了预训练权重初始化
   - 权重转换是否正确（TP/PP 配置是否一致）
4. **精度检查**：
   - 是否使用了 bf16 混合精度训练
   - 检查是否存在 NaN 或 Inf 值

### NPU 显存不足 (OOM)

**现象**：`RuntimeError: NPU out of memory`

**排查步骤**：

1. 减小 `micro-batch-size`（最低至 1）
2. 增大并行度：
   - 增大 TP（`--tensor-model-parallel-size`），注意 `tp_size <= num_key_value_heads`
   - 增大 PP（`--pipeline-model-parallel-size`），注意 TP×PP ≤ NPU 数量
3. 减小序列长度 `--seq-length`
4. 开启重计算：

   ```bash
   --recompute-granularity full --recompute-method block --recompute-num-layers <层数>
   ```

5. 使用 ChunkLoss 降低显存峰值（参考 [ChunkLoss](features/chunkloss.md)）
6. 使用分布式优化器 `--use-distributed-optimizer`

### LLM PP 切分为 0 层报错

**现象**：`AssertionError: learning_rate is None`

**排查步骤**：

1. 检查 PP 层数配置，确保 LLM 部分每个 stage 均有层数
2. 错误示例：`llm=[0,8,10,10]`（第一个 stage LLM 层数为 0）
3. 正确示例：`llm=[1,7,10,10]`（每个 stage 至少 1 层 LLM）

### tp_size 超过 num_key_value_heads

**现象**：`RuntimeError: tp_size exceeds num_key_value_heads`

**排查步骤**：

1. 查看模型配置中的 `num_key_value_heads` 值
2. 确保 `--tensor-model-parallel-size` 不超过该值
3. 如需更大并行度，可结合 PP 或 CP 策略

---

## 分布式训练故障排查

### 多机启动卡死

**现象**：多机多卡启动时脚本卡死，无报错信息。

**排查步骤**：

1. 检查是否安装了 `pdsh`：

   ```bash
   which pdsh
   ```

   如未安装：`apt install pdsh` 或 `yum install pdsh`
2. 检查节点间网络连通性：

   ```bash
   ping <MASTER_ADDR>
   ```

3. 检查端口是否被占用：

   ```bash
   netstat -tlnp | grep <MASTER_PORT>
   ```

### 通信超时

**现象**：`HCCL timeout`、`RuntimeError: Timeout`

**排查步骤**：

1. 检查网卡名称并设置环境变量：

   ```bash
   ifconfig
   export HCCL_SOCKET_IFNAME=<网卡名>
   export TP_SOCKET_IFNAME=<网卡名>
   export GLOO_SOCKET_IFNAME=<网卡名>
   ```

2. 增大通信超时时间：

   ```bash
   export HCCL_CONNECT_TIMEOUT=600
   ```

3. 检查各节点 CANN 版本是否一致

### 快慢卡负载不均衡

**现象**：训练效率低下，部分卡等待时间长。

**排查步骤**：

1. 使用 profiling 工具分析各卡计算时间（参考 [工具使用](tools.md)）
2. 检查是否启用了异构并行配置
3. 考虑使用多模态异构 PP 切分（参考 [异构并行](features/hetero_parallel.md)）
4. 考虑启用 encoder 数据负载均衡（参考 [Encoder数据负载均衡](features/encoder_dp_balance.md)）

---

## 权重转换故障排查

### HuggingFace 权重加载失败

**现象**：`KeyError`、`RuntimeError: shape mismatch`

**排查步骤**：

1. 确认已使用 `mm-convert` 工具转换权重（参考 [权重转换](features/mm_convert.md)）
2. 检查转换时的并行配置与训练脚本是否一致：
   - `tp_size`
   - `pp_size`
   - `llm_pp_layers` / `vit_pp_layers`
3. 检查权重文件是否完整下载

### 权重保存超时

**现象**：保存 checkpoint 时超时报错。

**排查步骤**：

1. 检查磁盘 IO 带宽是否正常
2. 确认磁盘空间是否充足
3. 可适当增大保存间隔 `--save-interval`
4. 该超时通常不影响已保存的权重，可忽略

---

## Docker 故障排查

### 容器内缺少模型依赖

**现象**：`ModuleNotFoundError` 运行模型时报错。

**排查步骤**：

1. Docker 镜像仅预装 torch、torch_npu 和 decord 基础依赖
2. 根据目标模型的 README 在 base 环境中手动安装额外依赖
3. 参考 [Docker 使用](../../docker/OVERVIEW.zh.md)

### CANN 版本冲突

**现象**：多版本 CANN 环境下训练异常。

**排查步骤**：

1. 建议使用 Docker 隔离不同 CANN 版本
2. 检查容器内环境变量是否指向正确版本
3. 避免在宿主机和容器间共享 CANN 安装路径

---

## 调试技巧

### 开启同步模式定位错误位置

当训练出现错误但无法定位具体位置时，可开启同步模式：

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

> **警告**：同步模式会严重影响性能，仅在 debug 时使用，生产环境必须关闭。

### 开启 NPU 特征值检测

```bash
export NPU_ASD_ENABLE=2
```

取值说明：

- `0`：关闭检测
- `1`：仅打印异常日志
- `2`：打印异常日志并告警
- `3`：打印异常日志、告警，并在 device 侧 info 日志中记录过程数据

### 日志级别调整

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0  # DEBUG，输出最详细日志
export ASCEND_SLOG_PRINT_TO_STDOUT=1  # 日志直接打印到终端
```

> **注意**：DEBUG 级别日志流量大，会影响训练性能，仅在排查问题时短暂开启。

### 使用 Profiling 工具

详细使用方法请参考 [工具使用](tools.md#profiling采集工具)。
