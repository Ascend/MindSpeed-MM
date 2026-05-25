# MindSpeed-MM Agent Configuration

This directory contains shared guidance for AI coding agents working on MindSpeed-MM.

The `.agents` directory is the single source for reusable agent-facing context. Tool-specific directories such as `.codex/`, `.claude/`, `.cursor/`, or `.trae/` can be generated locally from this shared source when needed.

MindSpeed-MM follows the [Agent Skills](https://agentskills.io/home) convention for skill layout.

## Directory Layout

| Path | Purpose |
| --- | --- |
| `skills/` | Skill index and implementation conventions. |
| `knowledge/` | Shared knowledge context for agents. |
| `setup_agent.sh` | Optional helper for linking `.agents` into local tool-specific directories. |

## Usage

Link this shared configuration into a local agent directory:

```bash
bash .agents/setup_agent.sh codex
bash .agents/setup_agent.sh claude
bash .agents/setup_agent.sh cursor
bash .agents/setup_agent.sh trae
```

The script also accepts a custom agent name and creates `.<agent-name>/` as a local adapter directory. Generated adapter directories are added to `.git/info/exclude`.

## Architecture Summary

MindSpeed-MM supports two main training backend paths. Agents should identify the active backend before changing model code, data code, checkpoint conversion, examples, or tests.

| Backend | Primary Entries | Description |
| --- | --- | --- |
| MindSpeed Core / Megatron | `mindspeed_mm/training.py`, `mindspeed_mm/pretrain_*.py`, `examples/*/*.sh` | Megatron-style flow using model/data/forward providers and hybrid parallelism. |
| FSDP2 | `mindspeed_mm/fsdp/train/trainer.py`, `mindspeed_mm/config/config_manager.py`, `mindspeed_mm/fsdp/utils/register.py` | YAML-driven flow using plugin registration, `ModelHub`, FSDP2 data builders, and parallel plans. |

See `knowledge/architecture.md` for the agent-facing architecture overview.

## Skill Domain

| Domain | Status | Description |
| --- | --- | --- |
| Integration | Planned | Onboarding workflows for models, data, environments, and checkpoints. |
| Verification | Planned | Validation workflows for correctness, tests, inference, and evaluation. |
| Data | Planned | Data-format checks for multimodal and tool-augmented training. |
| Optimization | Planned | Performance analysis and configuration guidance for training workloads. |
| Collaboration | Planned | Review, documentation, and contribution workflow assistance. |

See `skills/README.md` for the full skill index.
