# MindSpeed MM Documentation

MindSpeed MM is an Ascend multimodal large model suite for large-scale distributed training. It supports both multimodal generation and multimodal understanding, and aims to provide end-to-end multimodal training solutions for Huawei Ascend chips. It includes preset mainstream industry models, data engineering, distributed training and acceleration, and features such as pre-training, fine-tuning, online inference tasks, and reinforcement learning.

- **Key technical support**: provides core technical capabilities such as long-sequence and large-scale distributed training
- **Flexible model design and development**: supports flexible design and efficient assembly and development of multimodal large models and tasks
- **Rich data engineering**: shortens data preparation time and accelerates model training through efficient multimodal data preprocessing capabilities and acceleration mechanisms
- **Preset models ready out of the box**: a rich variety of high-performance preset models covering multimodal tasks such as image generation, video generation, image-text understanding, and speech models, with "out-of-the-box" capability that lowers the barrier to use and accelerates project delivery
- **Built on the high-performance Ascend foundation MindSpeed-Core**: based on the Ascend high-performance distributed acceleration library MindSpeed-Core, it provides rich parallelism, memory, communication, and computing optimization capabilities, with more affinity optimizations to enhance acceleration in multimodal scenarios

```{toctree}
:caption: QuickStart:
:maxdepth: 1

guides/installation/install_guide
introduction/quick_practice
```

```{toctree}
:caption: Development Guide:
:maxdepth: 1

introduction/overview
guides/development/fsdp2_model_migration_guide_old
guides/development/new_model_development
```

```{toctree}
:caption: Feature Documentation:
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
:caption: Configuration:
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
:caption: Tuning Guide:
:maxdepth: 1

guides/tuning/memory_tuning
guides/tuning/performance_tuning
```

```{toctree}
:caption: FAQ:
:maxdepth: 1

guides/troubleshooting/FAQ
```
