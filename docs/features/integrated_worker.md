## GRPO多进程训推共卡

### 技术概述
所有 worker 共用一组 pg，并在该 pg 上分别完成分布式进程初始化，在此情况下 vit worker 可自定义切分策略。

#### 具体实现
##### 注意事项
共置功能走**单卡多进程 HCCL 通信**，在**25.RC1及更新版本**的 CANN、HDK 上支持该功能，其以下的版本不能使用 *colocate_actor_and_vit* 参数。
##### 配置方法

**在examples/rl/configs中训练yaml文件的rl_config字段中添加：**

```yaml
# 开启 actor 与 vit 共置
use_integrated_worker: true
colocate_actor_and_vit: true

# 分别指定 actor 与 vit 的卡数
actor_resource:
    num_npus: 16   # actor 进程需要的卡数
vit_resource:
    num_npus: 16    # vit 进程需要的卡数
```

**此时可以独立配置 actor_config 与 vit_config 中的分布式并行切分策略：**
```yaml
actor_config:
    tensor_model_parallel_size: 4     # actor TP 切分
    pipeline_model_parallel_size: 4   # actor PP 切分

vit_config:
    tensor_model_parallel_size: 1     # vit TP 切分
    pipeline_model_parallel_size: 1   # vit PP 切分
```

在 **MindSpeed-MM/examples/rl/envs/runtime_env.yaml** 下配置以下环境变量：
```yaml
HCCL_HOST_SOCKET_PORT_RANGE: "60000-60050"
HCCL_NPU_SOCKET_PORT_RANGE: "61000-61050"
```