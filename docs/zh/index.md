# MindSpeed-MM 文档

MindSpeed-MM是面向大规模分布式训练的昇腾多模态大模型套件，同时支持多模态生成及多模态理解，旨在为华为 昇腾芯片 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、在线推理任务、强化学习等特性。

- **关键技术支持**：提供长序列及大规模分布式训练等核心技术能力
- **模型灵活设计与开发**：支持多模态大模型及任务灵活设计与高效组装开发
- **丰富的数据工程**：通过高效的多模态数据预处理能力及加速机制，缩短数据准备时间，加速模型训练
- **预置模型开箱即用**：丰富多样的高性能预置模型，覆盖图像生成、视频生成、图文理解、语音模型等多模态任务，具备“开箱即用”能力，降低使用门槛，加速项目落地
- **基于高性能昇腾底座MindSpeed-Core**：基于昇腾高性能分布式加速库MindSpeed-Core提供丰富的并行，内存，通信，计算优化能力，更多亲和优化，增强多模态场景加速能力

```{toctree}
:caption: QuickStart:
:maxdepth: 1

guides/installation/install_guide
introduction/quick_practice
```

```{toctree}
:caption: 开发指南:
:maxdepth: 1

introduction/overview
guides/development/fsdp2_model_migration_guide_old
guides/development/new_model_development
```

```{toctree}
:caption: 特性文档:
:maxdepth: 1

introduction/feature_overview
features/parallel/fsdp2_principle
features/parallel/hetero-parallel
features/parallel/sequence_parallel
features/memory/async_activation_offload
features/memory/online_data_balance
features/parallel/tensor_parallel
```

```{toctree}
:caption: 配置说明:
:maxdepth: 1

reference/configuration_overview
reference/model_configuration
reference/data_configuration
reference/training_arguments
reference/mcore-fsdp2_configuration
reference/tools_configuration
reference/environment_variables
```

```{toctree}
:caption: 调优指南:
:maxdepth: 1

guides/tuning/memory_tuning
guides/tuning/performance_tuning
```

```{toctree}
:caption: FAQ:
:maxdepth: 1

guides/troubleshooting/FAQ
```
