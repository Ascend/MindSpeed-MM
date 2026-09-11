# MindSpeed MM FSDP2 Model Migration (Using Qwen3-VL as an Example)

## Overview

This document uses **Qwen3-VL-30B-A3B (an MoE multimodal understanding model)** as an example to describe the complete process of migrating a new model and integrating it into the MindSpeed MM FSDP2 backend, and explains the rationale at each key decision point for reference when migrating other models.

This document is intended for researchers, engineers, and developers who need to integrate a new model into the FSDP2 backend. Readers are expected to:

- Have completed the installation of the Ascend environment and MindSpeed MM according to [Installation Guide](../pytorch/install_guide.md).
- Have basic knowledge of PyTorch training and model development and debugging.
- Understand the basic concepts of model migration, distributed training, and precision alignment.
- Have the target model already trainable on the source platform (e.g., GPU) with a saved loss baseline, which will serve as a starting point and a reference for accuracy alignment.

If you only need to run Qwen3-VL fine-tuning using the ready-made example rather than integrating a new model, refer to [Qwen3VL README](../../../examples/qwen3vl/README_v1.md).

## Source Model and Migration Target

### Source Model Structure

The reference implementation of Qwen3-VL resides in the [src/transformers/models/qwen3_vl_moe](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl_moe) directory of the Hugging Face Transformers repository. Its top-level class is `Qwen3VLMoeForConditionalGeneration`, which mainly consists of three parts: a vision encoder, a language model, and an output head. The structure is as follows:

```text
Qwen3VLMoeForConditionalGeneration
├── model.visual                      # Vision encoder (ViT)
│   ├── blocks.0 ~ blocks.N           # Visual Transformer layers
│   ├── merger                        # Visual feature merger module
│   └── deepstack_merger_list.{*}     # DeepStack multi-layer feature fusion
├── model.language_model              # Language model (MoE structure)
│   ├── embed_tokens
│   └── layers.0 ~ layers.M           # Each layer's mlp.experts are sparse experts
└── lm_head                           # Output head
```

It is recommended to first map out this module tree: the FSDP sharding plan, freezing configuration, and recomputation configuration all use these module paths. `parallel.fsdp_plan.apply_modules` in the sample [`examples/qwen3vl/qwen3vl_30B_config_v1.yaml`](../../../examples/qwen3vl/qwen3vl_30B_config_v1.yaml) is configured according to these paths and can be used as a reference.

### Migration Target: Plugin-Style FSDP2 Backend

MindSpeed MM integrates new models via the plugin-style FSDP2 backend: the training entry point is `mindspeed_mm/fsdp/train/trainer.py`, and training is driven by a single top-level YAML. This document follows this backend, using the launch script `finetune_qwen3vl_30B_v1.sh` and the configuration `qwen3vl_30B_config_v1.yaml`.

Once the target is clear, the migration work is divided into four parts, producing the following files in the repository:

| Work Item | Location | Qwen3VL Approach |
|---|---|---|
| Model integration | `mindspeed_mm/fsdp/models/qwen3vl/` | Copy the Hugging Face modeling code into the repository and modify it |
| Data integration | No new files | Reuse the generic `huggingface` dataset + built-in `collator` |
| Training configuration | `qwen3vl_30B_config_v1.yaml` | A single top-level YAML |
| Launch script | `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh` | Start the trainer via torchrun |

## Model Integration

### Core Files in Plugin-Style FSDP2

The essence of model integration is to make the model class identifiable to the framework's training pipeline, constructible, and subject to FSDP2 strategies. The table below lists the core files involved in the integration process. Understanding their responsibilities is helpful for following the subsequent steps (all located under `mindspeed_mm/fsdp/`):

| File | Responsibility | How to Use During Integration |
|---|---|---|
| `train/trainer.py` | Main training entry point | Launched by torchrun; loads YAML, imports plugins, builds models and data, and enters training loop |
| `utils/register.py` | Registries (`model_register` / `data_register`) and plugin import | Model/data classes are registered with `@model_register.register("<id>")`; the directories listed in `training.plugin` are recursively imported to trigger registration |
| `models/modelhub.py` | Model construction hub (`ModelHub.build`) | First attempts `AutoConfig.from_pretrained` for Transformers-style construction; if it fails, falls back to custom model construction |
| `params/*_args.py` (e.g., `model_args.py`, `training_args.py`) | Parameter definitions corresponding to each YAML section (pydantic dataclass) | Consult these before writing YAML fields (field names, defaults, validation) |
| `distributed/torch_parallelize.py` | Parallelism/sharding strategy (`ParallelApplier`) | Applies `fully_shard` to the model according to `parallel.fsdp_plan`/`ep_plan`, as well as TP, EP, prefetch, etc. |
| `features/apply_features.py` | Training feature application (`FeaturesApplier`) | Applies recomputation, ChunkLoss, activation offloading, etc. according to the `features` section |
| `train/train_engine.py` | Single-step training logic (`TrainEngine`) | Calls the model with `model(**batch, use_cache=False)` and reads `output.loss`; model forward must match this calling convention |
| `checkpoint/dcp_checkpointer.py` | Checkpoint read/write (`DistributedCheckpointer`) | Saves and loads training checkpoints in DCP format (meta init loading of DCP weights also uses this) |

### Choosing an Integration Approach

There are three approaches to model integration: custom model integration, Transformers model integration, and third-party model adaptation wrapper. **Qwen3VL falls into the second category: Transformers model integration**. The model class retains inheritance from Hugging Face `PreTrainedModel` and follows the Hugging Face construction and weight loading mechanisms.

It should be noted that, even within Transformers model integration, there are two engineering forms: when upstream models do not require internal logic changes, you can directly import the upstream class and register it as needed. However, Qwen3VL requires modifications **inside the model's `forward`**, which cannot be satisfied by the import approach. Therefore, `modeling_qwen3_vl_moe.py` is copied into the repository as a whole and then modified. Examples of required modifications include:

- Sequence parallelism (Ulysses/Ring CP): inserting sequence splitting and aggregation communication into the visual/text forward pass
- MoE load-balancing auxiliary loss (aux loss): computing the auxiliary loss in the forward pass and accumulating it into the total loss

The third approach, "third-party model adaptation wrapper" (an outer wrapper responsible for weight loading and input/output field adaptation), is suitable for scenarios where the source model structure is inconvenient to modify. However, it likewise cannot cover the internal forward modifications described above, so it is not adopted for Qwen3VL.

When selecting an integration approach for a new model, refer to the following table:

| Model Situation | Integration approach |
|---|---|
| Belongs to the Hugging Face ecosystem and requires no changes to internal logic | Transformers integration: directly import the upstream class and register as needed |
| Belongs to the Hugging Face ecosystem but requires in-depth forward modifications (Qwen3VL in this case) | Transformers integration: copy modeling into repository and modify, keep diff minimal for easier upstream upgrades |
| Not part of the Hugging Face ecosystem, or structure is hard to modify | Custom model integration, or third-party adaptation wrapper |

### Copying, Registration, and Framework Recognition

Model files should be placed under the designated directory. Qwen3VL has only two files:

```text
mindspeed_mm/fsdp/models/qwen3vl/
├── modeling_qwen3_vl_moe.py   # Copied from transformers and modified
└── npu_patch.py               # NPU fusion operator replacements
```

Integration with the framework only requires adding one line of registration decorator on the top-level class (in `modeling_qwen3_vl_moe.py`):

```python
from mindspeed_mm.fsdp.utils.register import model_register

@model_register.register("qwen3_vl_moe")
class Qwen3VLMoeForConditionalGeneration(Qwen3VLMoePreTrainedModel, GenerationMixin):
    ...
```

The registration name `"qwen3_vl_moe"` is the value to be filled in for `model.model_id` in the subsequent YAML. The framework imports the plugin directories listed in `training.plugin` to make the registration take effect.

Another key point is that the modified class **still inherits Hugging Face's `PreTrainedModel`**, so the framework automatically builds the model and loads weights in the Transformers manner, without the need to implement loading logic manually (only custom models outside the Hugging Face ecosystem need to implement `_from_config`/`from_pretrained`).

### Model Modifications: Mandatory and Optional Enhancements

The modifications in the repository can be divided into mandatory changes and optional enhancements. It is recommended to first complete the mandatory changes to get the model running in its most basic form, then add enhancements as needed rather than trying to complete everything upfront.

The mandatory changes are those that enable the model to be invoked by the training engine: `forward` receives the batch fields produced by the dataloader and returns an output object carrying `.loss` (the training engine invokes `model(**batch_data, use_cache=False)` and reads `output.loss`). Models inheriting from the Hugging Face's `PreTrainedModel` also need to set `accepts_loss_kwargs = False` (Qwen3VL already has this; when migrating other Hugging Face models, ensure it is retained).

Once the mandatory changes are complete and combined with the data and YAML configurations, the model can be trained normally. All other modifications are optional and only needed in specific scenarios: **Long-sequence training** requires adapting the model forward to handle sequence parallel splitting/aggregation communication; otherwise, keeping `parallel.ulysses_parallel_size: 1` is sufficient. For **performance optimization**, hot-spot operators can be replaced with the fused operators under `mindspeed_mm/fsdp/ops/`. When **expert load balancing is required for MoE models**, auxiliary loss must be supported. If your model does not involve these scenarios, they can be skipped; Qwen3VL involves all of the above categories, and its `modeling_qwen3_vl_moe.py` and `npu_patch.py` can serve as reference implementations.

## Data Integration

### Reuse and Adaptation of Data Processing

Raw data (dialogue JSON, images/videos) should be prepared according to the actual task. During migration, you need to determine whether the data processing pipeline can be reused or needs adaptation. The pipeline consists of three components, and reusability depends on the specific conditions:

- **Dataset construction** (`dataset_type`, responsible for loading raw data and performing preprocessing such as field mapping, conversation templating, and tokenization): If the data is in a standard conversation format, the general-purpose multimodal dataset registered as `huggingface` (`mindspeed_mm/fsdp/data/datasets/huggingface/`) is reused; if the format is non-standard (speech features, video latents, custom packing), a new dataset construction logic is registered using `@data_register.register`.
- **Multimodal packing plugin** (the `PLUGINS` table in `mindspeed_mm/fsdp/data/data_utils/func_utils/mm_plugin.py`, handling special tokens, visual/video token placeholder expansion, varying by model): If the model already exists in `PLUGINS`, reuse it; otherwise, add or override the corresponding plugin.
- **collator** (groups batches and produces training fields): Reuse an existing matching implementation; otherwise, implement `collate_fn` in the dataset or register a new collator in `data_collator.py`.

The actual choices for the three parts of Qwen3VL are as follows: for **dataset construction**, the generic `huggingface` is reused; for **multimodal packing**, `Qwen3VLPlugin` is used (it inherits `Qwen2VLPlugin` and overrides the visual token placeholders, video timestamps, and other aspects according to Qwen3-VL, expanding the `<image>`/`<video>` placeholders into sequences with vision special tokens and producing `pixel_values`/`image_grid_thw`); for **collator**, `DataCollatorForQwen2vl` is reused and registered as `qwen3vl` in `data_collator.py`.

### Data Integration Configuration

Once the dataset, plugin, and collator are in place, data integration is primarily reflected in the `data` section of the YAML file (where the `template` field selects the corresponding plugin). The key configuration for Qwen3VL is as follows (excerpt; paths are examples and should be modified according to your environment; full details are in `qwen3vl_30B_config_v1.yaml`):

```yaml
data:
  dataset_param:
    dataset_type: huggingface          # Corresponds to the dataset registration name
    attr:                              # Keys are framework concepts, values are the corresponding names in your data JSON (see explanation below)
      images: images
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
    preprocess_parameters:
      model_name_or_path: /home/data/Qwen3-VL-30B-A3B-Instruct   # Original Hugging Face directory, used to load the tokenizer/processor
      image_max_pixels: 262144         # Image resolution upper limit, which affects the number of visual tokens and video memory.
    basic_parameters:
      template: qwen3_vl_nothink       # Conversation template, which determines prompt formatting and must match the model.
      cutoff_len: 1024                 # Truncation length.
      dataset_dir: /home/usr/data/
      dataset: /home/usr/data/mllm_format_llava_instruct_data.json
      cache_dir: ./cache_dir/          # Preprocessing cache; do not share the same mount directory across multiple nodes.
  dataloader_param:
    sampler_type: BaseRandomBatchSampler
    collate_param:
      collator_id: qwen3vl              # Corresponds to the built-in collator name.
```

Regarding the direction of the `attr` mapping: **keys are fixed framework concept names, values are the corresponding names in your data**. `images`/`messages` are filled with the column names in your  data JSON, `role_tag`/`content_tag` are filled with the key names within each message, and `user_tag`/`assistant_tag` are filled with the values of the `role` field. In this example, each message in the data takes the form `{"role": "user", "content": "..."}`, so `role_tag: role` and `user_tag: user` are configured. If your data follows the classic sharegpt format (where each message takes the form `{"from": "human", "value": "..."}`), you should configure `role_tag: from`, `content_tag: value`, and `user_tag: human`.

Two common pitfalls:

- `model_name_or_path` is used here **only to load the tokenizer and processor**, not to load training weights (the weights are specified by `training.load`).
- An incorrect `template` will cause the prompt concatenation to differ from the model's pretraining format, resulting in normal loss decrease but poor final performance. Be sure to verify the template when migrating a new model.

## Training YAML Configuration

All training behavior of the plugin-style FSDP2 is driven by a single YAML file. The following sections describe the choices made for Qwen3VL; the complete fields can be found in `qwen3vl_30B_config_v1.yaml`.

### `parallel` Section: Sharding Plan

```yaml
parallel:
  tensor_parallel_size: 1              # Must be 1 in plugin mode
  fully_shard_parallel_size: auto      # Automatically set the FSDP sharding group based on the total number of devices
  fsdp_plan:
    apply_modules:
      - model.visual.blocks.{*}        # Shard each visual layer separately
      - model.visual.merger
      - model.visual.deepstack_merger_list.{*}
      - model.visual
      - model.language_model.embed_tokens
      - model.language_model.layers.{*}   # Shard each LLM layer separately
      - model.language_model
      - lm_head
    param_dtype: bf16
    reduce_dtype: fp32                 # FP32 used for gradient reduction to preserve precision
  expert_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts   # Reserved: shard by experts when EP is enabled
```

The values of `apply_modules` come from the module tree in [source model structure](#source-model-structure), where `{*}` is a wildcard for layer indices. The framework executes `fully_shard` on each module listed therein, as well as on the outermost model in sequence (when `apply_modules` is empty, only the outermost model is sharded). For regular fine-tuning, the sample configuration can be used directly. If customization is required, the module paths must be taken from the model's `named_modules()`, and when prefetch is enabled, do not arbitrarily reorder the module sequence in a proven configuration.

### `model` Section: Freezing

```yaml
model:
  model_id: qwen3_vl_moe               # Consistent with the name in @model_register.register
  model_name_or_path: /home/data/Qwen3-VL-30B-A3B-Instruct
  attn_implementation: flash_attention_2
  freeze:
    - model.visual                     # Freeze the visual encoder in fine-tuning scenarios
```

The modules listed in `freeze` are set to `requires_grad=False`, so they do not participate in training, and their gradients and optimizer states are no longer saved (saving corresponding memory). The sample freezes the visual encoder `model.visual`, If your task requires training the visual encoder, simply delete that line.

### `features` Section: Memory and Loss Strategy

```yaml
features:
  loss_cfg:
    loss_type: default
    router_aux_loss_coef: 0.0          # MoE auxiliary loss coefficient; only needed for MoE models; 0 disables it
  recompute: true
  recompute_plan:
    apply_modules:                     # Recomputation is enabled for both the visual and LLM parts
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}
```

The sample enables recomputation by default (`recompute: true`, trading computation for memory). These fields **must be placed in the top-level `features:` section**; placing them in the `model:` section does not raise an error but has no effect at all.

`model.visual.blocks` in `recompute_plan`: when the vision encoder is frozen, backward does not pass through it, so recomputation does not save memory there. It is kept so that the configuration does not need to be changed if the vision encoder is later trained; if you are certain you will not train it, you can remove it.

When memory is still tight, features such as ChunkLoss (`enable_chunk_loss`) and asynchronous activation offload (`enable_activation_offload`) can be enabled in the `features` section. The example YAML has corresponding configuration blocks reserved.

### `training` Section: Weight Loading

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 1       # If left empty, gradient accumulation is disabled; explicitly set a value when needed.
  lr: 1.0e-5
  lr_decay_style: cosine
  train_iters: 10000
  init_model_with_meta_device: true    # Enable for models with a large number of parameters.
  # load: <DCP weight directory>
  # save: <save directory>
  plugin:
    - mindspeed_mm/fsdp/models/qwen3vl          # Imported at startup to trigger model registration.
    - mindspeed_mm/fsdp/data/datasets/huggingface   # Trigger dataset registration.
```

Weight loading is a key decision in large-model migration, but why use meta init + DCP weights? Standard loading of large models incurs high peak memory during initialization. `init_model_with_meta_device` reduces it by first constructing the model structure on the meta device without allocating actual memory, then loading the sharded DCP format weights directly to each card — each card reads only its own partition (e.g., for Qwen3-VL-30B, the BF16 weights are about 60GB, avoiding repeated host memory usage across cards).

Meta init must be used together with DCP-format weights, so you must first convert HF weights to DCP before training (DCP is a sharded checkpoint format; conversion can be reused). Conversion command (run at the repository root, assuming HF weights are already downloaded to `ckpt/Qwen3-VL-30B-A3B-Instruct`):

```bash
mm-convert GenericDCPConverter hf_to_dcp \
  --hf_dir ckpt/Qwen3-VL-30B-A3B-Instruct \
  --dcp_dir ckpt/Qwen3-VL-30B-A3B-Instruct-dcp
```

Then uncomment `training.load` in the YAML and fill in the converted DCP directory `ckpt/Qwen3-VL-30B-A3B-Instruct-dcp`. For more usage of the conversion tool, see [Weight Conversion](../pytorch/weight_conversion.md).

The `plugin` list connects the outcomes of [model integration](#model-integration) and [data integration](#data-integration) into the framework: at launch, it imports these directories, models and datasets are registered, and the `model_id`/`dataset_type` can then find the corresponding implementations.

## Launch Script and Execution

The skeleton of the launch script `examples/qwen3vl/finetune_qwen3vl_30B_v1.sh` is as follows:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Modify according to the actual installation path.
export NON_MEGATRON=true            # Critical: select the plugin-style FSDP2 initialization path; must be set.
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1

NPUS_PER_NODE=16                    # Number of NPUs per node; modify according to the actual setup.
MASTER_ADDR=localhost               # Change to the primary node IP for multi-node deployment.
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3vl/qwen3vl_30B_config_v1.yaml
```

Note that the entry point of torchrun is the unified trainer `mindspeed_mm/fsdp/train/trainer.py`, and its only argument is the YAML file. This means that when migrating a new model, the launch script can be almost copied as-is, only changing the YAML path and the number of cards. The meaning of each environment variable is explained in comments within the script.

Adjust `NPUS_PER_NODE` according to the hardware in use. In the YAML, `fully_shard_parallel_size: auto` automatically determines the sharding group based on the total number of cards. When the number of cards is small, you can correspondingly reduce `micro_batch_size` and `cutoff_len`, or enable more memory optimizations to fit the available memory.

Once data and weights are ready, launch training at the repository root:

```bash
bash examples/qwen3vl/finetune_qwen3vl_30B_v1.sh
```

Logs are output to the `logs/` directory. Common operations such as weight download and COCO dataset preparation are not repeated here; follow the [Qwen3VL README](../../../examples/qwen3vl/README_v1.md) to perform them.

**How to confirm that training has started successfully**: After startup, the training log prints key metrics for each iteration at the `log_interval` frequency, in the following form:

```text
iteration 1/10000 | consumed samples: 8 | elapsed time per iteration (ms): 6603.7 | learning rate: 0.000000E+00 | global batch size: 8 | loss: 1.016570E+01 | grad norm: 50.001 |
iteration 2/10000 | consumed samples: 16 | elapsed time per iteration (ms): 2231.6 | learning rate: 1.000000E-08 | global batch size: 8 | loss: 1.009848E+01 | grad norm: 49.063 |
```

As long as the log keeps printing by iteration, `loss` stays within a reasonable range and generally decreases over training, and `grad norm` does not show NaN/Inf,training is running normally (the first iteration is usually slower due to compilation and initialization overhead, which is expected). If startup reports an error or hangs, refer to [FAQs](../FAQ.md) .

## After Training Starts Successfully

- **Accuracy alignment**: After the migrated model runs successfully, it is recommended to align its accuracy with the source repository (GPU/reference framework). The specific approach is to enable deterministic computation (`training.use_deter_comp: true`), fix the random seed, and disable data shuffle. After eliminating randomness, compare whether the loss curves of the two sides are consistent.
- **Performance tuning**: Collect profile data, locate bottlenecks, and enable sequence parallelism/prefetch/ChunkLoss as needed. See [Performance Tuning](../pytorch/performance_tuning.md).
- **Low-cost fine-tuning**: If memory is limited, switch to [LoRA Fine-tuning (FSDP2)](./lora_finetune_fsdp2.md).
- **Exporting weights**: The training output is in DCP format. Use `mm-convert GenericDCPConverter dcp_to_hf` to convert it back to Hugging Face format. See [Weight Conversion](../pytorch/weight_conversion.md).
  
This document uses Qwen3-VL as an example to walk through the complete migration process. For more comprehensive interface descriptions and field definitions of each configuration section, refer to the [FSDP2 Migration](./fsdp2_developer_migration_guide.md).
