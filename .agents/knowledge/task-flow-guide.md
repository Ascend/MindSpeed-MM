# Agent Task Workflow Guide

This document defines the standard task workflows for AI coding agents working on MindSpeed-MM. Use it to determine the correct sequence of actions, files to modify, and validation steps for common development tasks.

Always read `architecture.md` first to identify the target backend before following any workflow below.

---

## Quick Decision: Which Backend?

```text
User request mentions:
├── "FSDP2" / "YAML config" / "plugin" / "ModelHub" / "trainer.py"
│   → Use FSDP2 workflows (Section 1)
│
├── "Megatron" / "pretrain_" / "provider" / "training.py"
│   → Use Megatron workflows (Section 2)
│
├── "checkpoint" / "convert" / "hf2mg" / "mg2hf" / "DCP"
│   → Use Checkpoint workflows (Section 3)
│
├── "profile" / "FLOPs" / "MFU" / "memory"
│   → Use Tooling workflows (Section 4)
│
└── "test" / "CI" / "unit test"
    → Use Testing workflows (Section 5)
```

---

## 1. FSDP2 Task Workflows

### 1.1 Onboard a New VLM (Vision-Language Model)

This is the most common FSDP2 task. Follow the detailed skill at `skills/mindspeed-mm-fsdp2-model-only-vlm-migration/SKILL.md`.

**Quick checklist**:

| Step | Action | Key Files |
|------|--------|-----------|
| 1 | Read upstream model card, config, processor, tokenizer | `config.json`, `modeling_*.py`, `processing_*.py` |
| 2 | Record forward kwargs and confirm `labels` → `.loss` | Model `forward()` signature |
| 3 | Choose weight loading strategy (HF direct / DCP meta-init) | `modelhub.py` |
| 4 | Create model plugin at `mindspeed_mm/fsdp/models/<name>/` | `modeling_<name>.py`, `npu_patch.py` |
| 5 | Register with `@model_register.register("<model_id>")` | `register.py` |
| 6 | Decide data branch (A: reuse / B: partial / C: custom) | `qwen2vl_dataset.py`, `template.py`, `mm_plugin.py` |
| 7 | Align template, mm_plugin, collator, batch keys | `template.py`, `mm_plugin.py`, `data_collator.py` |
| 8 | Create `examples/<name>/` with YAML + launch script | `examples/<name>/<name>_config.yaml` |
| 9 | Fill `fsdp_plan.apply_modules` from `model.named_modules()` | YAML `parallel.fsdp_plan` |
| 10 | Verify: 1-step train runs without error | Launch script |

**Decision: Data Branch Selection**

```text
Can raw samples map to ShareGPT/Alpaca via DatasetAttr?
├── YES → Can existing template + mm_plugin + collator handle it?
│   ├── YES → Branch A: Only write YAML
│   └── NO  → Branch B: Add template/mm_plugin/collator, keep dataset_type=huggingface
└── NO  → Branch C: Write custom dataset plugin
```

### 1.2 Add a New Dataset to FSDP2

| Step | Action | Key Files |
|------|--------|-----------|
| 1 | Determine if Branch A/B/C applies | See decision tree above |
| 2 | Branch A: Configure `data.dataset_param.attr` in YAML | `<model>_config.yaml` |
| 3 | Branch B: Add template in `template.py` or mm_plugin in `mm_plugin.py` | `template.py`, `mm_plugin.py` |
| 4 | Branch B: Register new collator key in `data_collator.py` if needed | `data_collator.py` |
| 5 | Branch C: Create `mindspeed_mm/fsdp/data/datasets/<name>/` | New `*_dataset.py` |
| 6 | Branch C: Register with `@data_register.register("<dataset_type>")` | `register.py` |
| 7 | Update YAML: `dataset_type`, `attr`, `template`, `collate_param.model_name` | `<model>_config.yaml` |
| 8 | Verify: DataLoader produces correct batch keys | Print batch keys in 1-step run |

### 1.3 Write or Modify an FSDP2 YAML Config

**Required YAML sections**:

```yaml
model:           # model_id, model_name_or_path, parallel (TP/EP/CP plan)
data:            # dataset_param (type, attr, preprocess), dataloader_param (collate)
parallel:        # fsdp_plan (apply_modules, hook_modules, ignored_modules)
training:        # plugin list, train_iters, batch_size, lr, load/save paths
features:        # recompute, activation_offload, chunk_loss (optional)
tools:           # profiler, memory_profile (optional)
```

**Validation checklist**:

| Check | How to Verify |
|-------|---------------|
| `model.model_id` matches `@model_register` | grep for `register("<model_id>")` |
| `data.dataset_type` matches `@data_register` | grep for `register("<dataset_type>")` |
| `training.plugin` imports all model/data packages | Check import paths exist |
| `parallel.fsdp_plan.apply_modules` uses real module names | Print `model.named_modules()` |
| `parallel.tensor_parallel_size` is 1 (FSDP2 convention) | YAML value |
| All paths use local directories (not HF Hub at runtime) | Check `model_name_or_path`, `dataset` |

### 1.4 Debug an FSDP2 Training Failure

**Systematic debugging order**:

```text
1. Plugin import errors
   → Check training.plugin paths exist and are importable
   → Verify @register decorators executed (add print in register function)

2. Model build errors
   → Check model_id matches registration
   → Verify model_name_or_path contains valid config.json
   → For custom models: check _from_config / from_pretrained implementation

3. Data pipeline errors
   → Check dataset_type is registered
   → Verify attr mapping matches raw JSON field names
   → Check template name is registered or tokenizer has chat_template
   → Verify collate_param.model_name is a registered DATA_COLLATOR key

4. Parallel apply errors
   → Check apply_modules match real module names (case-sensitive)
   → Verify EP plan modules are subset of FSDP plan modules
   → Check TP size is 1 (FSDP2 convention)

5. Forward errors
   → Print batch keys from dataloader
   → Check model forward(**batch) accepts all keys
   → Verify labels produce loss when loss_type=raw

6. OOM errors
   → Reduce batch_size or enable activation_offload
   → Check recompute plan covers large modules
   → Enable chunk_loss for large vocab models
```

---

## 2. Megatron Task Workflows

### 2.1 Onboard a New Model (Megatron Backend)

| Step | Action | Key Files |
|------|--------|-----------|
| 1 | Create model implementation in `mindspeed_mm/models/<name>/` | `modeling_<name>.py` |
| 2 | Register model provider | `mindspeed_mm/pretrain_*.py` |
| 3 | Register data provider | `mindspeed_mm/data/` |
| 4 | Create example scripts in `examples/<name>/` | `*.sh` |
| 5 | Add Megatron argument parsing if needed | `mindspeed_mm/arguments.py` |

### 2.2 Modify Megatron Training Flow

| Change Type | Primary Files |
|-------------|---------------|
| Forward step logic | `mindspeed_mm/training.py` |
| Pretrain entry point | `mindspeed_mm/pretrain_*.py` |
| Model provider | `mindspeed_mm/models/` |
| Data provider | `mindspeed_mm/data/` |
| Loss function | `mindspeed_mm/loss/` |

---

## 3. Checkpoint Conversion Workflows

### 3.1 Determine Conversion Path

```text
Source format → Target format:
├── HuggingFace → DCP (FSDP2 training)
│   → checkpoint/fsdp/generic_dcp_converter.py
│
├── DCP → HuggingFace (inference / release)
│   → checkpoint/fsdp/generic_dcp_converter.py (dcp_to_hf)
│
├── HuggingFace → Megatron (PTD training)
│   → checkpoint/ptd/
│
├── Megatron → HuggingFace
│   → checkpoint/ptd/
│
└── Custom format
    → checkpoint/common/converter.py (base class)
```

### 3.2 Add a New Checkpoint Converter

| Step | Action |
|------|--------|
| 1 | Identify source and target formats |
| 2 | Extend `checkpoint/common/converter.py::Converter` |
| 3 | Implement key mapping (source → target parameter names) |
| 4 | Handle special cases: tied weights, MoE expert reshape, DTensor |
| 5 | Add conversion script in `examples/<model>/` |

---

## 4. Tooling Workflows

### 4.1 Profile Training Performance

| Step | Action | Key Files |
|------|--------|-----------|
| 1 | Enable profiler in YAML: `tools.profiler` | `<model>_config.yaml` |
| 2 | Set level: 0 (light) / 1 (medium) / 2 (heavy) | `tools_args.py` |
| 3 | Run training, collect trace | `profiler.py` |
| 4 | Analyze: `python -m mindspeed_mm.fsdp.tools.profiler analyse <trace>` | `profiler.py` |
| 5 | Identify bottleneck: DataLoader / H2D / Compute / Communication | Chrome trace |

### 4.2 Calculate FLOPs / MFU

| Step | Action | Key Files |
|------|--------|-----------|
| 1 | Identify model config: layers, hidden_size, num_heads, vocab_size | `config.json` |
| 2 | Use existing FLOPs tool or extend for new model | `tools/flops_tool/` |
| 3 | Calculate theoretical FLOPs per step | Formula in tool |
| 4 | Measure actual step time from training log | Training log |
| 5 | MFU = theoretical_FLOPs / (step_time × device_peak_FLOPs) | - |

### 4.3 Memory Profiling

| Step | Action |
|------|--------|
| 1 | Enable in YAML: `tools.memory_profile` |
| 2 | Set `record_peak_memory: true` |
| 3 | Run training, check memory report |
| 4 | Identify peak memory modules |
| 5 | Apply recompute / offload to peak modules |

---

## 5. Testing Workflows

### 5.1 Add Unit Tests

| Step | Action | Key Files |
|------|--------|-----------|
| 1 | Identify test scope: model / data / parallel / checkpoint | `tests/` |
| 2 | Create test file: `tests/fsdp/models/test_<name>.py` | `tests/` |
| 3 | Follow existing test patterns (see `tests/fsdp/models/test_patch_mimo_v2.py`) | `tests/` |
| 4 | Test minimal: model build + 1 forward pass | - |
| 5 | Test data: dataset build + 1 batch iteration | - |
| 6 | Run: `python -m pytest tests/fsdp/models/test_<name>.py -xvs` | - |

### 5.2 Run CI Locally

```bash
# FSDP2 unit tests
python -m pytest tests/fsdp/ -xvs -k "<test_name>"

# All tests
python -m pytest tests/ -xvs
```

---

## 6. Cross-Cutting Rules

### 6.1 File Naming Conventions

| Component | Pattern | Example |
|-----------|---------|---------|
| Model implementation | `modeling_<name>.py` | `modeling_qwen3_vl_moe.py` |
| NPU patches | `npu_patch.py` | `npu_patch.py` |
| Model config YAML | `<name>_<size>_config.yaml` | `qwen3vl_30B_config_v1.yaml` |
| Launch script | `finetune_<name>_<size>.sh` | `finetune_qwen3vl_30B_v1_A5.sh` |
| Test file | `test_<name>.py` | `test_patch_mimo_v2.py` |
| Dataset plugin | `<name>_dataset.py` | `qwen2vl_dataset.py` |

### 6.2 Import Conventions

```python
# FSDP2 imports
from mindspeed_mm.fsdp.utils.register import model_register, data_register
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.train.trainer import Trainer
from mindspeed_mm.fsdp.train.train_engine import TrainEngine
from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier
from mindspeed_mm.fsdp.features.apply_features import FeaturesApplier
from mindspeed_mm.fsdp.data import build_mm_dataset, build_mm_dataloader

# Megatron imports
from mindspeed_mm.training import pretrain
from mindspeed_mm.arguments import parse_args

# Common imports
from mindspeed_mm.fsdp.utils.device import get_device_type, get_torch_device
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
```

### 6.3 When to Create New Files vs Modify Existing

| Scenario | Action |
|----------|--------|
| New model (FSDP2) | Create `mindspeed_mm/fsdp/models/<name>/` |
| New model (Megatron) | Create `mindspeed_mm/models/<name>/` |
| New dataset type | Create `mindspeed_mm/fsdp/data/datasets/<name>/` |
| New template | Add to `template.py` |
| New mm_plugin | Add to `mm_plugin.py` |
| New collator | Add to `data_collator.py` |
| New parallel strategy | Modify `torch_parallelize.py` |
| New feature (recompute/offload) | Create in `features/` + register in `apply_features.py` |
| Bug fix in existing model | Modify existing file |
| Configuration change | Modify YAML or `*_args.py` |

### 6.4 Common Pitfalls

| Pitfall | Prevention |
|---------|------------|
| Modifying Megatron code for FSDP2 task | Check `architecture.md` backend decision tree first |
| Using HF Hub paths in YAML | Always use local paths for `model_name_or_path` and `dataset` |
| Guessing `apply_modules` names | Print `model.named_modules()` and copy exact names |
| Forgetting to register | Every model needs `@model_register`, every dataset needs `@data_register` |
| Missing `training.plugin` entry | YAML must list all packages with `@register` decorators |
| Enabling EP/prefetch/offload too early | First bring-up should use minimal features |
| Ignoring batch key compatibility | Print batch keys and verify model `forward` accepts them all |

---

## 7. Task-to-File Mapping (Quick Reference)

| User Request | Start Here |
|-------------|------------|
| "Add model X to FSDP2" | `skills/.../SKILL.md` → `mindspeed_mm/fsdp/models/` → `examples/` |
| "Add dataset Y" | `mindspeed_mm/fsdp/data/datasets/` → `data_collator.py` → YAML |
| "Fix training OOM" | `features/memory/` → YAML `features` section |
| "Convert checkpoint A→B" | `checkpoint/` → `converter.py` |
| "Profile training" | `tools/profiler.py` → YAML `tools` section |
| "Add parallel strategy" | `distributed/torch_parallelize.py` → `parallel_args.py` |
| "Fix forward/loss" | `mindspeed_mm/fsdp/models/<name>/modeling_*.py` |
| "Fix data loading" | `data/datasets/` → `data_collator.py` → `convert.py` |
| "Add unit test" | `tests/fsdp/models/` |
| "Update docs" | `docs/zh/features/` or `examples/<name>/README.md` |

---

## 8. Validation Checklist Before Submitting PR

| Check | Command / Method |
|-------|-----------------|
| Model builds | 1-step training run |
| Data loads | Print batch keys from dataloader |
| Forward produces loss | Check `output.loss` is scalar |
| Backward completes | 1 optimizer step without error |
| Config is valid | YAML parses without error |
| Plugin imports | No `ModuleNotFoundError` for plugin packages |
| No hardcoded paths | All paths from YAML config |
| NPU compatible | No CUDA-only API calls (use `device.py` wrappers) |
| Code style | Follow existing patterns in same directory |
