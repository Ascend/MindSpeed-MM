# 环境变量

## 适用后端

该文档适用于Mcore、 Mcore-FSDP2后、FSDP2后端。

常用环境变量的描述如下：

<div class="container">
        <table class="mm-table">
            <thead>
                <tr>
                    <th>环境变量</th>
                    <th>描述</th>
                    <th>取值说明</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code class="env-var">ASCEND_SLOG_PRINT_TO_STDOUT</code></td>
                    <td>是否开启日志打印</td>
                    <td>
                        <code class="value-desc">0</code>: 关闭日志打屏<br>
                        <code class="value-desc">1</code>: 开启日志打屏
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">ASCEND_GLOBAL_LOG_LEVEL</code></td>
                    <td>设置应用类日志的日志级别</td>
                    <td>
                        <code class="value-desc">0</code>: DEBUG<br>
                        <code class="value-desc">1</code>: INFO<br>
                        <code class="value-desc">2</code>: WARNING<br>
                        <code class="value-desc">3</code>: ERROR<br>
                        <code class="value-desc">4</code>: NULL（不输出日志）
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">TASK_QUEUE_ENABLE</code></td>
                    <td>控制task_queue算子下发队列优化等级</td>
                    <td>
                        <code class="value-desc">0</code>: 关闭<br>
                        <code class="value-desc">1</code>: 开启Level 1优化<br>
                        <code class="value-desc">2</code>: 开启Level 2优化
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">COMBINED_ENABLE</code></td>
                    <td>优化非连续两个算子组合场景</td>
                    <td>
                        <code class="value-desc">0</code>: 关闭<br>
                        <code class="value-desc">1</code>: 开启
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">CPU_AFFINITY_CONF</code></td>
                    <td>控制CPU任务绑核行为</td>
                    <td>
                        <code class="value-desc">0</code>或未设置: 不绑核<br>
                        <code class="value-desc">1</code>: 粗粒度绑核<br>
                        <code class="value-desc">2</code>: 细粒度绑核
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">HCCL_CONNECT_TIMEOUT</code></td>
                    <td>设备间socket建链超时时间</td>
                    <td>
                        整数 [120,7200]<br>
                        <code class="value-desc default-value">默认：120</code> (秒)
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">PYTORCH_NPU_ALLOC_CONF</code></td>
                    <td>控制缓存分配器行为</td>
                    <td>
                        <code class="value-desc">expandable_segments:</code>: 启用内存池扩展段
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">HCCL_EXEC_TIMEOUT</code></td>
                    <td>设备间执行同步等待时间</td>
                    <td>
                        整数 [68,17340]<br>
                        <code class="value-desc default-value">默认：1800</code> (秒)
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">ACLNN_CACHE_LIMIT</code></td>
                    <td>Host侧缓存的算子信息条目数</td>
                    <td>
                        整数 [1, 10,000,000]<br>
                        <code class="value-desc default-value">默认：10000</code>
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">TOKENIZERS_PARALLELISM</code></td>
                    <td>控制Hugging Face分词器并行行为</td>
                    <td>
                        <code class="value-desc">False</code>: 禁用并行<br>
                        <code class="value-desc">True</code>: 启用并行
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">MULTI_STREAM_MEMORY_REUSE</code></td>
                    <td>控制多流内存复用</td>
                    <td>
                        <code class="value-desc">0</code>: 关闭<br>
                        <code class="value-desc">1</code>: 开启
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">NPU_ASD_ENABLE</code></td>
                    <td>控制特征值检测功能</td>
                    <td>
                        <code class="value-desc">0</code>或未设置: 关闭<br>
                        <code class="value-desc">1</code>: 开启，仅打印异常<br>
                        <code class="value-desc">2</code>: 开启并告警<br>
                        <code class="value-desc">3</code>: 开启并记录过程数据
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">ASCEND_LAUNCH_BLOCKING</code></td>
                    <td>控制算子执行模式</td>
                    <td>
                        <code class="value-desc">0</code>: 异步执行<br>
                        <code class="value-desc">1</code>: 同步执行
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">NPUS_PER_NODE</code></td>
                    <td>配置节点使用的NPU数量</td>
                    <td>
                        整数，如 <code class="value-desc">1</code>, <code class="value-desc">8</code>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>

<div style="height: 40px;"></div>

> **备注：** 更多昇腾相关环境变量请参考 [环境变量列表](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0001.html)
