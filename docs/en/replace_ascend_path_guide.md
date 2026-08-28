# MindSpeed MM HDK Installation Path Batch Replacement Guide

## Background

Some installation documents in the MindSpeed MM repository use the hard-coded `/usr/local/Ascend/driver` path. 
If the actual installation path of the HDK you are using is `/usr/local/npu/driver`, you need to complete the batch replacement before use to ensure that environment variables can be loaded properly.

This guide provides the complete steps for batch path replacement using the `replace_ascend_path.py` script.

---

## Prerequisites

- Python 3.7+
- Read and write permissions for the repository directory
- It is recommended to commit or back up the current state via `git` before performing the replacement.

---

## Affected File Scope

| File Type | Description | Typical Path Example |
|---------|------|-------------|
| Shell scripts (`.sh`) | Training/testing startup scripts | `examples/*/pretrain_*.sh`, `scripts/install.sh` |
| Markdown documents (`.md`) | Installation guides, model usage instructions | `docs/zh/pytorch/install_guide.md`, `docker/OVERVIEW.md` |
| RST documents (`.rst`) | User guides | `UserGuide/quick_start/environment_setup.rst` |
| Python files (`.py`) | Source code (if path references exist) | Source files of each module |
| Dockerfile | Docker image build scripts | `docker/Dockerfile` |

> Path variant notes: The repository contains the following references to Ascend/driver paths, all of which will be replaced together:
>
> - `/usr/local/Ascend/driver/` (Docker mount path)
> - `/usr/local/Ascend/driver/lib64/` (Docker mount path)
> - `/usr/local/Ascend/driver/version.info` (Docker mount path)

---

## Usage Steps

### Step 1: Enter the Repository Root Directory

```bash
cd /path/to/MindSpeed-MM
```

### Step 2: Preview the Changes to be Made (Recommended)

Before making actual modifications, first confirm the scope of changes in `--dry-run` mode:

```bash
python3 scripts/replace_ascend_path.py --dry-run
```

Example output:

```bash
[DRY RUN] Path replacement: /usr/local/Ascend/driver -> /usr/local/npu/driver
Scan directory : /path/to/MindSpeed-MM
File types     : .md, .py, .rst, .sh + Dockerfile
------------------------------------------------------------
Found XXX candidate file(s), processing...

  [would replace  12] docker/OVERVIEW.md
  [would replace  12] docker/OVERVIEW.zh.md
  ...

============================================================
[DRY RUN] XXX file(s) would be modified, XXX replacement(s) total.
          Remove --dry-run to apply changes.
```

### Step 3: Perform Batch Replacement

After confirming that the preview is correct, perform the actual replacement:

```bash
# Default: replace /usr/local/Ascend/driver with /usr/local/npu/driver
python3 scripts/replace_ascend_path.py
```

After execution, the script outputs the number of modified files and the total number of replacements.

### Step 4: Verify the Replacement Result

```bash
# Check whether any unreplaced paths remain.
grep -r "/usr/local/Ascend/driver" . \
  --include='*.sh' --include='*.md' --include='*.rst' --include='*.py' \
  --exclude-dir='.git' | wc -l
```

---

## Post-Execution Verification

### 1. Environment Variable Verification

```bash
# Verify that NPU is available.
npu-smi info
```

### 2. Core Feature Smoke Testing

Refer to the README of the corresponding model for installation and configuration, and verify that the training process can start normally.

```bash
# Source the CANN script (use the actual installation path).
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Run the example script (use the specific model).
bash examples/<model_name>/pretrain_<model_name>.sh
```

---

## Script Parameter Description

```bash
usage: replace_ascend_path.py [-h] [--source SOURCE] [--target TARGET]
                               [--dir DIR] [--extensions EXT [EXT ...]]
                               [--dry-run]

Options:
  -h, --help            Display help information
  --source SOURCE       Source path (default: /usr/local/Ascend/driver)
  --target TARGET       Target path (default: /usr/local/npu/driver)
  --dir DIR             Directory to scan (default: current directory .)
  --extensions EXT...   File extension whitelist (default: .sh .md .rst .py)
  --dry-run             Preview changes only, without modifying files
```
