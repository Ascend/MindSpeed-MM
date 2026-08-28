# MindSpeed MM FSDP2 Migration

This document is intended for developers who need to integrate new models, new datasets, or third-party training pipelines into the MindSpeed MM FSDP2 backend. Using the new pluginable training backend  `mindspeed_mm/fsdp` within the repository as the primary focus, this document describes the development conventions for model integration, data integration, YAML configuration, startup scripts, weight loading, and running.

MindSpeed MM FSDP2 is built on PyTorch FSDP2. On this basis, MindSpeed MM adds capabilities for the Ascend platform, including parallel state management, model registration, data registration, DCP checkpointing, recomputation, LoRA, expert parallelism, and multimodal data processing.

Migration work is typically divided into the following parts: model integration, data integration, configuration and startup scripts, and running.

## 1. Identify the FSDP2 Route

Two easily confused FSDP2 usage methods coexist in the MindSpeed MM repository. They differ in training entry points, configuration files, and model/data integration methods. Before migration, you must first determine the target route. This document focuses on the new plugin-style FSDP2; new models and datasets are recommended to be integrated via this route. The Megatron-bridged FSDP2 is a transitional form, mainly maintained for legacy entry point compatibility, and will no longer be the direction for new feature iteration. For the legacy route, please continue to refer to `docs/en/features/fsdp2.md` and the corresponding `examples/*/fsdp2_config.yaml`.

| Item | New plugin-Style FSDP2 (Main Focus) | Megatron-Bridged FSDP2 (Legacy Route) |
|---|---|---|
| Common entry point | `mindspeed_mm/fsdp/train/trainer.py`, or task-specific entry points under `mindspeed_mm/fsdp/tasks/*` | `pretrain_transformers.py`, `pretrain_vlm.py`, `pretrain_omni.py`, etc. |
| Configuration form | A single top-level YAML containing top-level configuration blocks such as `parallel`, `data`, `model`, `features`, `training`, and `tools` | Main training configuration + an additional `fsdp2_config.yaml` |
| Enabling method | The startup script directly passes the top-level YAML; model, data, and other plugins are imported through `training.plugin` | Command-line passes `--use-torch-fsdp2` and `--fsdp2-config-path` |
| Model integration | `model_register` + `ModelHub.build` | `model_provider`, `forward_step`, `loss_func` |
| Data integration | `data_register` + `build_mm_dataset/build_mm_dataloader` | `train_valid_test_datasets_provider`, `get_batch` |
| Sharding scale field | `parallel.fully_shard_parallel_size` | `sharding_size` |
| Submodule sharding field | `parallel.fsdp_plan.apply_modules` | `sub_modules_to_wrap` |

The table above is only for route identification, to avoid mixing entry points and configuration fields between the two paths.

The new plugin-style pipeline can be summarized as follows:

```text
torchrun
  -> mindspeed_mm/fsdp/train/trainer.py
  -> ConfigManager loads the top-level YAML
  -> training.plugin recursively imports plugins and triggers registration
  -> ModelHub.build builds the model
  -> LoRA (optional) injection
  -> TP/EP/Recompute/FSDP2 strategies are applied
  -> build_mm_dataset / build_mm_dataloader builds data
  -> TrainEngine: model(**batch_data, use_cache=False).loss
```

## 2. Pre-Migration Preparation

Before starting development, it is recommended to prepare the following information.

| Preparation Item | What to Confirm |
|---|---|
| Source repository entry point | Positions where model definition, weight loading, dataset construction, and training script are located respectively |
| Runtime assets | Model weight path, tokenizer/processor path, real training samples, and the root directories for image/audio/video/features |
| Model I/O | Which training batch fields `forward` requires, and whether it directly returns `.loss` |
| Weight format | Whether to load directly from Hugging Face/third-party weights, or to convert to DCP weights for meta initialization |
| Parallelism requirements | Whether capabilities such as CP, EP, recomputation, prefetch, activation offload, and LoRA are needed |
| Reference case | Find a case of the same modality or construction approach under`examples/<case_name>` as a baseline |

It is recommended to start from existing integrated case entry points in the repository. The table below serves only as an entry index for quickly locating similar models; specific field reuse and route identification should be based on actual scripts, YAMLs, and model code.

| Type | Case | Entry Point and Configuration |
|---|---|---|
| Standard training entry point `trainer.py`/VLM | Qwen3.5 Dense | `examples/qwen3_5/finetune_qwen3_5_{4B,9B,27B}.sh`, `qwen3_5_{4B,9B,27B}_config.yaml` |
| Standard training entry point `trainer.py`/MoE VLM | Qwen3.5-MoE, Qwen3.6 | `examples/qwen3_5/qwen3_5_{35B,122B,397B}_config.yaml`, `examples/qwen3_6/qwen3_6_35B_A3B_config.yaml` |
| Standard training entry point `trainer.py`/VLM | Qwen3VL v1 | `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh`, `qwen3vl_30B_config_v1.yaml` |
| Standard training entry point `trainer.py`/omni-modal model | Qwen3Omni v1 | `examples/qwen3omni/finetune_qwen3omni_v1.sh`, `qwen3omni_config_v1.yaml` |
| Standard training entry point `trainer.py`/custom VLM | KimiK2.5 | `examples/kimik2_5/finetune_kimik2_5.sh`, `kimik2_5_config.yaml` |
| Standard training entry point `trainer.py`/video/audio generation model | LTX2 | `examples/ltx2/finetune_ltx2_t2v.sh`, `finetune_ltx2_t2av.sh`, `ltx2_config_t2v.yaml`, `ltx2_config_t2av.yaml` |
| Standard training entry point `trainer.py`/speech synthesis | Qwen3TTS | `examples/qwen3tts/finetune_qwen3tts.sh`, `qwen3tts_config.yaml` |
| Task-specific entry point/speech recognition | FunASR | `examples/funasr/finetune_funasr.sh`, `mindspeed_mm/fsdp/tasks/funasr/trainer.py`, `funasr_config.yaml` |
| Task-specific entry point/speech generation | CosyVoice3 | `examples/cosyvoice3/finetune_cosyvoice3.sh`, `mindspeed_mm/fsdp/tasks/cosyvoice3/train.py`, `cosyvoice3_config.yaml` |

## 3. Model Integration

Model adaptation is usually placed in:

```text
mindspeed_mm/fsdp/models/<model_name>/
```

The goal is to enable `training.plugin` to import the plugin, enable `model.model_id` to locate the model class, and allow the model to be built by `ModelHub.build` and subsequently processed by the FSDP2 strategy.

### 3.1 Three Common Integration Methods

| Method | Applicable Scenario | Development Focus |
|---|---|---|
| Custom model integration | The model body is maintained within the repository, or the source model can be refactored in the MM manner | Inherit `BaseModel` and implement `_from_config` and `from_pretrained` |
| Transformers model integration | The model is designed based on the Hugging Face `PreTrainedModel` | Keep the Hugging Face's `from_pretrained` signature compatible, and register the model class when necessary |
| Third-party model wrapper | The source model structure is not suitable for major modification | An outer adapted wrapper class is responsible for loading, field adaptation, and `.loss` output |

`ModelHub.build` first attempts to call `AutoConfig.from_pretrained(model.model_name_or_path)`. If successful, it typically follows the Transformers-style build path. If it fails, it falls back to the custom model build path and invokes `from_pretrained(ModelArguments)`.

### 3.2 Minimal Interfaces for Custom Models

```python
import torch

from mindspeed_mm.fsdp.models.base_model import BaseModel
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.utils.register import model_register


@model_register.register("<model_id>")
class XxxForTraining(torch.nn.Module, BaseModel):
    @classmethod
    def _from_config(cls, config: ModelArguments):
        # Only build the model structure from the configuration. This must be implemented if meta device initialization is enabled.
        ...

    @classmethod
    def from_pretrained(cls, config: ModelArguments):
        # Load weights from config.model_name_or_path, config.checkpoint_path, or a custom field.
        ...

    def forward(self, **batch):
        # TrainEngine calls model(**batch_data, use_cache=False) and reads output.loss.
        ...
```

Key points:

- In the custom model path, the full `ModelArguments` object is passed to `from_pretrained`, rather than a standalone path string.
- `_from_config` must build the complete module structure to support `training.init_model_with_meta_device: true`.
- `forward` must be able to receive the batch fields produced by the dataloader and be compatible with `**kwargs`.
- When `features.loss_cfg.loss_type: raw` is set, the model output must contain `.loss`. If the source model returns a tuple or dict, it is recommended to wrap it into an object with a `.loss` attribute.
- MoE auxiliary loss requires native model support. If the MoE model is copied from Transformers code, confirm that the original model already supports auxiliary loss computation. Refer to `Qwen3_5MoeForConditionalGeneration.overwrite_transformer_config` in `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py` to overwrite the transformer config. Also ensure that the router logits to capture are configured in `_can_record_outputs` and that the relevant modules are correctly using Transformers' `capture_outputs`.
- Logic such as special tokens, embedding resize, and `config.use_cache=False` should only be added when truly necessary for source model training. Avoid introducing untrackable behavioral differences at the migration layer.

## 4. Data Integration

Data adaptation is usually placed in:

```text
mindspeed_mm/fsdp/data/datasets/<dataset_or_model_name>/
```

The goal is to enable `data.dataset_param.dataset_type` in the configuration to locate the data construction logic. The registered object can be either a dataset class or a factory function.

### 4.1 Minimal Dataset Interfaces

The framework invokes the registered object with three parameters: `basic_param`, `preprocess_param`, and `dataset_param`. Therefore, the dataset constructor in the source repository usually requires a layer of adaptation.

```python
from mindspeed_mm.fsdp.utils.register import data_register


@data_register.register("<dataset_type>")
class XxxDataset:
    def __init__(self, basic_param, preprocess_param, dataset_param=None, **kwargs):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...

    def collate_fn(self, features):
        ...
```

Or:

```python
@data_register.register("<dataset_type>")
def build_xxx_dataset(basic_param, preprocess_param, dataset_param=None, **kwargs):
    return XxxDataset(...)
```

### 4.2 Dataloader and `collate` Conventions

The `collate` selection rules are as follows:

1. If the dataset object implements a callable `collate_fn`, the dataset's own `collate_fn` is used first.
2. Otherwise, a built-in collator, such as `qwen3vl`, `qwen3omni`, and `llm_pretrain` is looked up from `DATA_COLLATOR` based on `dataloader_param.collate_param.model_name`.
3. For custom datasets, if the batch format is special, it is preferable to implement `collate_fn` in the dataset to avoid polluting the general collator.

### 4.3 Training Batch Fields

`TrainEngine` first moves the batch to the current device and then executes it. 
Therefore, the batch keys must match the input parameters of the model's `forward`. For example, for `forward(input_ids, labels, pixel_values, **kwargs)`, the data side must produce fields with the same names in the dataset or `collate_fn`.

Batch field adaptation is often required during migration because the data sample format, field naming, and training invocation method of the source repository are not necessarily consistent with those of MindSpeed MM. Common adaptations include:

- Renaming source data fields to the names accepted by the model's `forward`.
- In multimodal tasks, organizing text, images, audio, video, or precomputed features into a structure that the model can read directly.
- Keep the batch as flat a dict as possible; the current `move_to_device` only handles top-level tensors, tensor lists, primitive types, and `None`. Complex nested structures must be handled in the `collate` function or at the model entry point.
- Floating-point tensors are moved to the device and cast according to `parallel.fsdp_plan.param_dtype`. Integer tensors remain as integers.

Common field examples: language models commonly use `input_ids/labels/attention_mask`, image-text models commonly use `pixel_values/image_grid_thw/image_flags`, speech models commonly use `speech_feat/speech_token/text_token`, and video generation models commonly use `video_latent/prompt_embeds/timesteps`. The specific fields are determined by the current model's `forward` method.

## 5. YAML Configuration

The plugin-style FSDP2 uses a single top-level YAML, usually placed in:

```text
examples/<model_name>/<model_name>_config.yaml
```

The following is a basic skeleton. Refer to similar YAML files to remove or extend fields for models in use.

```yaml
parallel:
  tensor_parallel_size: 1
  fully_shard_parallel_size: auto
  fsdp_plan:
    apply_modules:
      - model.language_model.layers.{*}
    # Recommended when EP or FSDP prefetch is enabled; can be removed as needed for non-MoE models.
    hook_modules:
      - model.language_model.layers.{*}
    param_dtype: bf16
    reduce_dtype: fp32
  ring_attention_size: 1
  ulysses_parallel_size: 1
  expert_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts

data:
  dataset_param:
    dataset_type: <dataset_type>
    preprocess_parameters:
      model_name_or_path: <tokenizer_or_processor_path>
    basic_parameters:
      dataset_dir: <data_root>
      dataset: <train_data_path>
  dataloader_param:
    pin_memory: true
    shuffle: true
    dataloader_mode: sampler
    drop_last: true
    sampler_type: BaseRandomBatchSampler
    num_workers: 4
    collate_param:
      model_name: <collate_name>

model:
  model_id: <model_id>
  model_name_or_path: <model_path>
  trust_remote_code: true
  freeze: []

features:
  loss_cfg:
    loss_type: raw
  recompute: false
  recompute_plan:
    apply_modules:
      - model.language_model.layers.{*}

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  seed: 42
  lr: 1.0e-5
  lr_decay_style: cosine
  lr_warmup_ratio: 0.1
  weight_decay: 0.0
  train_iters: 10
  clip_grad: 1.0
  init_model_with_meta_device: false
  optimizer: adamw
  adam_fused: true
  save_interval: 1000
  load: null
  save: null
  use_deter_comp: false
  plugin:
    - mindspeed_mm/fsdp/models/<model_name>
    - mindspeed_mm/fsdp/data/datasets/<dataset_or_model_name>

tools:
  profile:
    enable: false
  memory_profile:
    enable: false
```

Key consistency relationships:

- `model.model_id` must be consistent with `@model_register.register("<model_id>")`.
- `data.dataset_param.dataset_type` must be consistent with `@data_register.register("<dataset_type>")`.
- `training.plugin` must include the model plugin and data plugin paths. The plugin path may use `/`, which is converted to a Python package path upon import.
- `parallel.fsdp_plan.apply_modules` uses the module path pattern from the model's `named_modules()`. When prefetch is enabled, do not arbitrarily adjust the module order in a verified configuration.

Important fields:

Parallel strategy fields:

| Field | Description |
|---|---|
| `parallel.fully_shard_parallel_size` | FSDP sharding group size. `auto` derives it from `world_size // tensor_parallel_size`. |
| `parallel.tensor_parallel_size` | The current plugin-style FSDP2 code requires this to be `1`; setting it to a non-`1` value triggers a validation error. |
| `parallel.fsdp_plan.apply_modules` | Specifies the submodules to be wrapped by `fully_shard`. The framework then also calls `fully_shard` on the outermost model. When empty, only the outermost model is wrapped. |
| `parallel.fsdp_plan.hook_modules` | Specifies the modules where the FSDP hook manager is attached. When EP is enabled, it should be configured at a stable upper-level hierarchy, such as `model.language_model.layers.{*}`; otherwise expert-layer communication and prefetching can easily cause memory pressure. |
| `parallel.fsdp_plan.cpu_offload` | Offloads FSDP parameters and other states to CPU. When enabled, the initialization and communication backends also take CPU-related paths, which should be validated against memory and performance. |
| `parallel.expert_parallel_size`/`ep_plan.apply_modules` | MoE EP configuration. Enable it only when the model's expert modules have been adapted for EP. |

Data fields:

| Field | Description |
|---|---|
| `data.dataset_param.dataset_type` | Data registration name, which must be consistent with `@data_register.register("<dataset_type>")`. |
| `data.dataset_param.preprocess_parameters` | Preprocessing parameters such as tokenizer, processor, sampling, and truncation; the specific fields are read by the dataset implementation. |
| `data.dataset_param.basic_parameters` | Basic data parameters such as data root directory, data files, templates, and cache; the specific fields are read by the dataset implementation. |
| `data.dataloader_param.collate_param.model_name` | Name of the built-in collator; if the dataset provides its own `collate_fn`, this field is not used preferentially. |

Model fields:

| Field | Description |
|---|---|
| `model.model_id` | Model registration name, which must be consistent with `@model_register.register("<model_id>")`. |
| `model.model_name_or_path` | Path to Hugging Face/third-party weights, config, or the local model directory. |
| `model.freeze` | Freezes parameters by module path pattern. |

Feature fields:

| Field | Description |
|---|---|
| `features.loss_cfg.loss_type` | Defaults to `raw`, indicating that the `.loss` output by the model is used directly. |
| `features.recompute`/`features.recompute_plan.apply_modules` | Recomputation configuration, trading computation for memory; module paths likewise come from `named_modules()`. |

Training and checkpoint fields:

| Field | Description |
|---|---|
| `training.micro_batch_size` / `gradient_accumulation_steps` | Per-rank micro batch size and gradient accumulation steps. When `gradient_accumulation_steps` is empty, the framework disables gradient accumulation. |
| `training.init_model_with_meta_device` | Whether to first build the model structure on the meta device, used to reduce peak initialization memory for large models. |
| `training.load`/`save` | Weight/checkpoint loading and saving paths; when empty, the corresponding action is not executed. `load` can point to a DCP directory or an Hugging Face safetensors directory. |
| `training.plugin` | Plugin paths for models, data, etc. that need to be imported. |

Tool fields:

| Field | Description |
|---|---|
| `tools.profile` | Profiling configuration. |
| `tools.memory_profile` | Memory snapshot configuration. |

When `fully_shard_parallel_size: 1` is set and meta initialization is not used, the framework falls back to DDP wrapping, which facilitates small-scale debugging.

## 6. Weight Loading and Checkpoint

The plugin-style FSDP2 uses `DistributedCheckpointer` by default, which saves and loads DCP-format state based on `torch.distributed.checkpoint`.

### 6.1 Common Loading Methods

| Scenario | Recommended Method |
|---|---|
| The model can be loaded directly from Hugging Face/third-party weights, and single-server memory is sufficient | Set `training.init_model_with_meta_device: false`, and load the original weights in the model's `from_pretrained`. |
| Load Hugging Face weights online | Set `training.init_model_with_meta_device: true`, and configure the original weight path in `training.load: <hf_dir>`. |
| For large models, first convert to DCP format offline before loading | First use `hf_to_dcp` for conversion; set `training.init_model_with_meta_device: true` and `training.load: <dcp_dir>`. |
| Resume training from a checkpoint saved by the framework | Set `training.load`, and configure `no_load_optim`, `no_load_rng`, and `load_strict` as needed. |

Key semantics of meta initialization:

- When `init_model_with_meta_device: true` is set, the model structure is first built on the meta device, and then parameters are initialized according to `training.load`.
- If `training.load` is not empty, the framework moves the parameters to the target device or the CPU offload device, and then the DCP loads the state.

Save-related fields:

- `training.save`: root directory for checkpoint saving; when empty, no saving is performed.
- `training.save_interval`: saves at iteration intervals.
- `training.no_save_optim`/`training.no_save_rng`: controls whether to save the optimizer and random number states.
- `training.no_load_optim`/`training.no_load_rng`: controls whether to restore the optimizer and random number states.
- `training.load_strict`: passed to the DCP load planner. During migration debugging, it can be relaxed temporarily, but for formal training, strict matching should be maintained as much as possible.

## 7. Startup Script

The startup script is usually placed in:

```text
examples/<model_name>/finetune_<model_name>.sh
```

For existing cases, it is recommended to start directly from the corresponding script; for new models, refer to the following skeleton:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NON_MEGATRON=true
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/<model_name>/<model_name>_config.yaml
```

Common environment variables and their functions:

| Variable | Function |
|---|---|
| `NON_MEGATRON=true` | Selects the non-Megatron initialization path, adapting to the plugin-style FSDP2 entry point. |
| `HCCL_CONNECT_TIMEOUT` | Sets the timeout for establishing connections between HCCL devices, in seconds; this is usually increased in multi-device or multi-server scenarios. |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | Configures the torch-npu cache allocator to enable expandable memory segments, alleviating memory fragmentation issues in large model training. |
| `MULTI_STREAM_MEMORY_REUSE` | Controls the multi-stream memory reuse strategy; it is recommended to reuse values already validated on similar models. |
| `TASK_QUEUE_ENABLE` | Controls the optimization level of the task queue operator dispatch queue, with common values of `0/1/2`. |
| `CPU_AFFINITY_CONF` | Controls the CPU-side task core binding strategy, reducing fluctuations caused by task scheduling and NUMA access. |

Meanings of distributed startup variables:

| Variable | Description |
|---|---|
| `NPUS_PER_NODE` | Number of NPUs on the current node participating in training, which also corresponds to `torchrun --nproc_per_node`. |
| `NNODES` | Total number of nodes participating in training. |
| `NODE_RANK` | Rank of the current node. It is usually `0` for a single node, and ranges from `0` to `NNODES-1` across multiple nodes. |
| `MASTER_ADDR` | Address of the primary node. In a multi-node scenario, it is usually set to the IP of the node with `NODE_RANK=0`. |
| `MASTER_PORT` | Communication port of the primary node. Select a port that is not occupied on the current server. |
| `WORLD_SIZE` | Total number of processes, generally equal to `NPUS_PER_NODE * NNODES`. |

For multi-node training, the same `MASTER_ADDR`, `MASTER_PORT`, and `NNODES` must be set on each node, and a different `NODE_RANK` must be set for each node.

## 8. Running

After completing the development of the model, data, YAML configuration, and startup script described above, you can start training with `bash examples/<model_name>/finetune_<model_name>.sh`.
