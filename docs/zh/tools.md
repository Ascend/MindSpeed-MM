# MindSpeed MM 工具库使用指南

<p align="left">
        <b>简体中文</b> |
</p>

- [MindSpeed MM 工具库使用指南](#mindspeed-mm-工具库使用指南)
  - [Profiling采集工具](#profiling采集工具)
    - [静态采集](#静态采集)
    - [动态采集](#动态采集)
  - [Sora类模型特征提取](#sora类模型特征提取)
  - [内存快照提取](#内存快照提取)
  - [Tensorboard使用](#tensorboard使用)

## <a id="jump1"></a>Profiling采集工具

套件集成了昇腾[profiling采集工具](../../mindspeed_mm/tools/profiler.py)，以提供对模型运行情况的分析。内置模型均已适配，只需修改[tools.json](../../mindspeed_mm/tools/tools.json)文件即可生效。

【若新增模型，请先适配如下设置】

```python
from mindspeed_mm.tools.profiler import Profiler

prof = Profiler(prof_config)
prof.start()
while train:
    train_one_step
    prof.step()
prof.stop()
```

【通用的模型config设置】

```bash
--enable                  # 指开启profiling采集
--profile_type            # 指动态或静态的profiling采集类型, static / dynamic
--ranks                   # 指profiling采集的rank, default 为-1， 指采集全部rank
```

### 静态采集

`Static Profile`静态采集功能为执行模型训练过程中的指定的steps区间进行采集, 操作步骤如下：

1. 在模型config设置里开启`enable`采集开关，设置`profile_type` 为 static, 设置 `ranks`

2. 配置静态采集的相关参数

    【静态采集的参数设置/`static_param`】

    ```bash
    --level                     # profiling采集的level选择: level0, level1, level2
    --with_stack                # 采集时是否采集算子调用栈
    --with_memory               # 采集时是否采集内存占用情况
    --record_shapes             # 采集时是否采集算子的InputShapes和InputTypes
    --with_cpu                  # 采集时是否采集CPU信息
    --save_path                 # profiling的保存路径
    --start_step                # 设置启动采集的步数
    --end_step                  # 设置结束采集的步数
    --data_simplification       # 采集时是否采用简化数据
    --aic_metrics_type          # 采集模式，目前支持PipeUtilization和ArithmeticUtilization两种，默认采用PipeUtilization
    --analyse_flag              # 是否启用在线解析
    ```

3. 运行模型并采集profiling文件

4. 解析profiling文件

`analyse_flag`为`true`时，将在模型运行过程中自动执行解析。

`analyse_flag`为`false`时，生成的profiling文件位于`save_path`路径下，需要搭配如下命令触发离线解析：

```shell
    python mindspeed_mm/tools/profiler.py
```

### 离线解析命令参数说明

```shell
--mm-tool <path>               # MM工具配置文件路径，默认为./mindspeed_mm/tools/tools.json
--profiler-path <path>         # Profiler数据目录路径（可选，若不指定则从配置文件中读取）
--max-process-number <number>  # 分析的最大进程数（可选，默认：CPU核心数/2）
--export-type <type>           # 分析结果的导出类型，支持：text、db，可多次指定，默认：text
```

**示例：**
```shell
# 使用配置文件中的路径进行离线解析
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json

# 指定Profiler数据目录路径
python mindspeed_mm/tools/profiler.py --profiler-path ./npu_profiling

# 导出多种格式的分析结果（多次指定--export-type参数）
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json --export-type text --export-type db
```

此命令将解析指定路径下的所有profiling数据。参考 [离线解析](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/Profiling/atlasprofiling_16_0034.html)。

对超长序列、超大模型、强化学习等profiling文件较大的场景，使用离线解析可以节约训练时资源占用。

### 动态采集

`Dynamic Profile`动态采集功能可在执行模型训练过程中随时开启采集进程，操作步骤如下：

1. 在模型config设置里开启`enable`采集开关，设置`profile_type` 为 dynamic, 设置 `ranks`

2. 配置动态采集的相关参数

    【动态采集的参数设置`dynamic_param`】

    ```bash
    --config_path               # config与log文件的路径
    ```
  
    - `config_path`指向空文件夹并自动生成`profiler_config.json`文件
    - `config_path`指已有动态配置文件`profiler_config.json`的路径

3. 运行模型

4. 在模型运行过程中，随时修改`profiler_config.json`文件配置，profiling采集会在下一个step生效并开启

    【动态采集的实现方式】

    - 动态采集通过识别`profiler_config.json`文件的状态判断文件是否被修改，若感知到`profiler_config.json`文件被修改，`dynamic_profile`会在下一个step时开启Profiling任务
    - `config_path`目录下会自动记录`dynamic_profile`的维测日志

动态采集的具体参数、入参表、及具体操作步骤等请[参考链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/devtools/profiling/atlasprofiling_16_0033.html#ZH-CN_TOPIC_0000002046667974__section17272160135118)

## <a id="sorafeature"></a>Sora类模型特征提取

[feature_extraction](../../mindspeed_mm/tools/feature_extraction)目录下工具可用于提取视频和文本特征并保存，目前支持单batch静态数据集特征提取。按需修改[tools.json](../../mindspeed_mm/tools/tools.json)文件。

```bash
--extract_video_feature # 是否提取视频特征
--extract_text_feature  # 是否提取文本特征
--save_path             # 特征数据存储路径
```

使用前按需修改[feature_extraction_t2v.sh](./feature_extraction/feature_extraction_t2v.sh)文件中对应模型数据集和配置文件（VAE、T5）路径。

```bash
--MM_DATA       # 数据配置文件路径(.json)
--MM_MODEL      # 模型配置文件路径(.json)
```

配置完成后，调用[feature_extraction_t2v.sh](./feature_extraction/feature_extraction_t2v.sh)即可提取数据特征。

## <a id="memory"></a>内存快照提取

套件集成了昇腾[内存快照采集工具](../../mindspeed_mm/tools/mem_profiler.py)，以提供对模型运行情况的分析。内置模型均已适配，只需修改[tools.json](../../mindspeed_mm/tools/tools.json)文件即可生效。

对复用[训练流程](../../mindspeed_mm/training.py)的模型，同样仅需修改配置。支持的配置项如下。

```json5
{
  "memory_profile": {
    "enable": false,    // 内存采集功能开关
    "start_step": 0,    // 开始录制的步数。数值为训练步数的起始点，0代表初始化过程
    "end_step": 2,      // 结束录制的步数。数值为训练步数的起始点，0代表初始化过程
    "save_path": "./memory_snapshot",  // 快照文件保存路径
    "dump_ranks": [     // 录制快照的rank列表，从0开始
      0
    ],
    "stacks": "all",    // 堆栈信息录制。可选项：python/all
    "max_entries": null, // 最大记录数，null则无限制
    "mem_info": false    // 是否记录内存信息
  }
}
```

对独立的训练流程，可参考下列代码，对训练脚本做适配以使用profiler特性。参数配置同上。

```python
from megatron.training import get_args
from mindspeed_mm.tools.mem_profiler import memory_profiler

args = get_args()                                   # 获取配置
memory_profiler.reset(args.mm.tool.memory_profile)  # 使用配置刷新profiler状态
training_preparation()                              # 运行训练准备代码
while iteration < args.train_iters:                 # 训练主循环
    memory_profiler.step()                          # 调用profiler记录一个迭代
    train_one_step()                                # 训练一个迭代
memory_profiler.stop()                              # 停止采集
```

对于不具备典型训练结构的脚本，或者局部的手动调试，可直接调用基础函数。(不推荐)

```python
code_not_record()
from mindspeed_mm.tools.mem_profiler import _record
_record()
code_to_record()
```

dump与开始录制可以在不同文件内。

```python
code_to_record()
from mindspeed_mm.tools.mem_profiler import _dump, _stop
_dump()
_stop()
```

dump执行完成后，会在输出目录生成`snapshot_`开头的`pickle`文件，可以在[torch页面](https://pytorch.org/memory_viz)可视化查看内存快照。

## <a id="Tensorboard使用"></a>Tensorboard使用

1. 若使用tensorboard，需进行安装：

    ```bash
    pip install tensorboard
    ```

2. 设置tensorboard的保存路径`TENSORBOARD_LOGS_PATH`，然后在运行脚本中的`OUTPUT_ARGS`中添加`--tensorboard-dir`从而进行使能：

    ```shell
    TENSORBOARD_LOGS_PATH="./tensorboard_dir/" # tensorboard保存路径

    # 在args里添加 --tensorboard-dir
    OUTPUT_ARGS="
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --load $LOAD_PATH \
        --save $SAVE_PATH \
        --ckpt-format torch \
        --tensorboard-dir $TENSORBOARD_LOGS_PATH \ 
    "
    ```

3. 打开tensorboard进行查看，`./tensorboard_dir/`为step2中的保存路径，按需修改

    ```shell
    tensorboard --logdir ./tensorboard_dir/
    ```
