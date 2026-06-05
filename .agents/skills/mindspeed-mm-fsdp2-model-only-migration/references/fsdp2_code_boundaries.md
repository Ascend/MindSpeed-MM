# MindSpeed-MM FSDP2 框架地图

当迁移新模型时，用本参考按需理解 FSDP2 关键链路、文件职责和改动分级。

## 核心训练链路

```text
examples/<model_id>/*.sh
  -> mindspeed_mm/fsdp/train/trainer.py
  -> training.plugin 导入模型/数据插件
  -> ModelHub.build
  -> ParallelApplier(TP/FSDP2)
  -> build_mm_dataset
  -> build_mm_dataloader
  -> TrainEngine.train_step
  -> model(**batch_data, use_cache=False).loss
  -> backward / optimizer step
```

## 重要文件树

```text
mindspeed_mm/fsdp/
├── train/
│   ├── trainer.py                  # 训练入口：解析 YAML，导入 training.plugin，构建模型、数据、optimizer 与 TrainEngine
│   └── train_engine.py             # 训练执行器：batch 上设备，调用 model(**batch).loss，执行 backward、optimizer step、日志与保存调度
├── models/
│   ├── modelhub.py                 # 模型构建中枢：根据 model.model_id 找 registry，区分 HF config 路径和 custom BaseModel 路径
│   ├── base_model.py / mtp.py       # BaseModel 兼容层与多 token prediction 辅助
│   ├── qwen3vl/                    # 重点理解类参考：Qwen 系 VLM/MoE modeling，适合看 HF 风格模型注册、NPU patch、视觉输入到语言模型的接线
│   ├── kimik2_5/                   # 重点理解类参考：Kimi/DeepSeek modeling + media_utils，适合看模型目录内如何保留上游主体并做局部适配
│   ├── qwen3omni/                  # 全模态理解/生成模型参考
│   ├── qwen3tts/ / cosyvoice3/     # TTS/语音生成模型参考
│   ├── funasr/                     # ASR/语音理解模型参考
│   ├── ltx2/                       # 视频/多模态生成模型参考
│   ├── internvlu/                  # VLM + diffusion 组合模型参考
│   └── <model_id>/                 # 新模型插件目录：configuration、modeling、processor、vision/audio、本地最小 patch
├── data/
│   ├── __init__.py                 # 数据入口：build_mm_dataset 根据 dataset_type 找 data_register，build_mm_dataloader 创建训练 dataloader
│   ├── dataloader/
│   │   ├── data_collator.py        # collator 注册表：DATA_COLLATOR key 到 collator class/function 的映射，承接最终 batch key 组装
│   │   ├── dataloader.py           # DataLoader 构建：读取 dataset/collate 配置，实例化 collate_fn、sampler 和 torch DataLoader
│   │   └── sampler.py              # 分布式 sampler：处理 rank/world size 下的数据切分和迭代顺序
│   ├── data_utils/func_utils/
│   │   ├── template.py             # chat/template 层：注册对话模板，处理角色壳、assistant 收尾、tool/thinking 文本和模态占位 token 位置
│   │   ├── mm_plugin.py            # 多模态协议层：校验/展开媒体占位符，组织 image/video/audio 输入，桥接 processor 输出字段
│   │   ├── convert.py              # 样本转换层：将 ShareGPT/Alpaca 类样本映射为训练特征，串联 template、mm_plugin 和 tokenizer
│   │   ├── collator.py             # 通用 collator 基类：处理 padding、labels、attention mask、媒体 fake input 与多模态 batch 公共行为
│   │   ├── formatters.py           # prompt/message 格式化层：把标准化 message 转成 template 可消费的片段
│   │   └── utils.py / model_args.py / log.py
│   └── datasets/
│       ├── huggingface/qwen2vl_dataset.py  # huggingface dataset_type 实现：本地 JSON + DatasetAttr + SupervisedDatasetProcessor 数据链路
│       ├── funasr/ / cosyvoice3/ / qwen3tts/ / ltx2/  # 特定任务数据插件参考：语音、TTS、预计算特征等非标准样本链路
│       └── <dataset_or_model_id>/           # 模型专用 dataset plugin 目录
├── distributed/                    # TP/FSDP2/EP/CP 并行实现：进程组、parallel plan、fully shard、expert/context parallel
├── features/                       # 训练高级特性：recompute、offload、chunk loss、prefetch、chunk_mbs 等
├── params/                         # YAML 参数 schema：model/data/training/parallel/feature/lora/tools 参数定义
├── checkpoint/                     # checkpoint / DCP 保存加载框架
├── optimizer/                      # optimizer、lr scheduler、grad norm 构建
├── loss/                           # 框架 loss function 构建和注入
├── ops/                            # NPU patch / fused op / 算子适配
├── tasks/                          # 少数任务专用 trainer/engine
├── tools/                          # profiler、debug、flops 工具
└── utils/                          # register、plugin import、device、dtype、LoRA 等通用工具

examples/<model_id>/                # 新模型配置区：YAML、启动脚本
```

## 改动分级

| 级别 | 范围 | 要求 |
| --- | --- | --- |
| 可以改 | `mindspeed_mm/fsdp/models/<model_id>/` | 新模型源码、配置类、processor/vision/audio 辅助和最小 patch 放这里；优先复用 HF 模型目录或开源仓代码。 |
| 可以改 | `examples/<model_id>/` | 新模型 YAML、启动脚本、最小 E2E 配置和样例说明放这里。 |
| 可以改 | `mindspeed_mm/fsdp/data/datasets/<dataset_or_model_id>/` | 只有现有 ShareGPT/Alpaca/huggingface 链路无法表达真实样本时，才新增 dataset plugin。 |
| 可以小改 | `template.py`、`mm_plugin.py`、`data_collator.py` | 只新增模型专用注册项、字段映射或 collator key；不得改变已有模型行为。 |
| 不建议改 | `convert.py`、`collator.py`、`formatters.py`、`qwen2vl_dataset.py`、已有无关模型/example、`params/` schema | 这些是共享链路或既有样例；优先用模型局部适配解决。 |
| 不能改 | `trainer.py`、`train_engine.py`、`modelhub.py`、`data/__init__.py`、`dataloader.py`、`sampler.py` | 这是 FSDP2 主训练与数据构建流程；模型接入不应触碰。 |
| 不能改 | `distributed/`、`checkpoint/`、`optimizer/`、`ops/`、`utils/register.py` | 这是框架核心；除非用户明确要求修框架 bug，否则不要为模型迁移修改。 |
