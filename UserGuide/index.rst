.. MindSpeed-MM documentation master file, created by
   sphinx-quickstart on Mon Dec  1 21:59:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpeed-MM 文档
==========================

MindSpeed-MM是面向大规模分布式训练的昇腾多模态大模型套件，同时支持多模态生成及多模态理解，旨在为华为 昇腾芯片 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、在线推理任务、强化学习等特性。

- **关键技术支持**：提供长序列及大规模分布式训练等核心技术能力
- **模型灵活设计与开发**：支持多模态大模型及任务灵活设计与高效组装开发
- **丰富的数据工程**：通过高效的多模态数据预处理能力及加速机制，缩短数据准备时间，加速模型训练
- **预置模型开箱即用**：丰富多样的高性能预置模型，覆盖图像生成、视频生成、图文理解、语音模型等多模态任务，具备“开箱即用”能力，降低使用门槛，加速项目落地
- **基于高性能昇腾底座MindSpeed-Core**：基于昇腾高性能分布式加速库MindSpeed-Core提供丰富的并行，内存，通信，计算优化能力，更多亲和优化，增强多模态场景加速能力

.. toctree::
   :maxdepth: 1
   :caption: QuickStart:

   quick_start/环境搭建
   quick_start/快速实践

.. toctree::
   :maxdepth: 1
   :caption: 开发指南:

   dev_guide/框架结构介绍
   dev_guide/模型迁移
   dev_guide/新模型开发

.. toctree::
   :maxdepth: 1
   :caption: 特性文档:

   features/特性总览
   features/fsdp2
   features/hetero-parallel
   features/sequence-parallel
   features/async-offload
   features/online-data-balance
   features/tensor-parallel

.. toctree::
   :maxdepth: 1
   :caption: 配置说明:

   config/配置概览
   config/模型配置
   config/数据配置
   config/训练参数
   config/fsdp2配置
   config/工具配置
   config/环境变量

.. toctree::
   :maxdepth: 1
   :caption: 调优指南:

   tuning/显存调优
   tuning/性能调优
   tuning/调优案例

.. toctree::
   :maxdepth: 1
   :caption: API参考:

   api/overview

.. toctree::
   :maxdepth: 1
   :caption: FAQ:

   faq/常见问题
