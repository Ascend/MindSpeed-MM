# Replacing CANN Installation in MindSpeed MM Scripts

## Background

In the overseas version, the default installation path for CANN/HDK has been changed from `/usr/local/Ascend` to `/usr/local/npu`.
The training scripts, installation documents, and other files in the MindSpeed MM repository use the hard-coded `/usr/local/Ascend` path.
If the actual installation path on the overseas machine is `/usr/local/npu`, a batch replacement must be performed to ensure that environment variables can be loaded normally.

This guide provides the complete steps for performing batch path replacement using the `replace_ascend_path.py` script.

## Prerequisites

- Python 3.7+
- Read and write permissions for the repository directory
- It is recommended to commit or back up the current state via git before performing the replacement.

## Scope of Affected Files

| File Type | Description | Typical Path Example |
|---------|------|-------------|
| Shell scripts (`.sh`) | Training/testing launch scripts | `examples/*/pretrain_*.sh`, `scripts/install.sh` |
| Markdown documents (`.md`) | Installation guides, model usage instructions | `docs/en/install_guide.md`, `docker/OVERVIEW.md` |
| RST documents (`.rst`) | User guides | `UserGuide/quick_start/environment_setup.rst` |
| Python files (`.py`) | Source code (if path references exist) | Source files in each module |
| Dockerfile | Docker image build scripts | `docker/Dockerfile` |

> Path Variants: The following Ascend path references exist in the repository and will all be replaced:
>
> - `/usr/local/Ascend/cann/set_env.sh` (most common; environment variable initialization)
> - `/usr/local/Ascend/ascend-toolkit/set_env.sh` (Ascend Toolkit initialization)
> - `/usr/local/Ascend/nnal/atb/set_env.sh` (ATB library initialization)
> - `/usr/local/Ascend/driver/lib64/` (Docker mount path)

## Usage Instructions

### Step 1: Enter the Repository Root Directory

```bash
cd /path/to/MindSpeed-MM
```

### Step 2: Preview the Changes to Be Made (Recommended)

Before making modifications, first confirm the scope of changes using `--dry-run`.

```bash
python3 scripts/replace_ascend_path.py --dry-run
```

Output example:

```bash
[DRY RUN] Path replacement: /usr/local/Ascend -> /usr/local/npu
Scan directory : /path/to/MindSpeed-MM
File types     : .md, .py, .rst, .sh + Dockerfile
------------------------------------------------------------
Found XXX candidate file(s), processing...

  [would replace   1] UserGuide/dev_guide/model_development.rst
  [would replace   1] UserGuide/quick_start/Qwen3VL-30B-MoE_fine-tuning_practice.rst
  ...

============================================================
[DRY RUN] XXX file(s) would be modified, XXX replacement(s) total.
          Remove --dry-run to apply changes.
```

### Step 3: Execute Batch Replacement

After confirming the preview is correct, proceed with the replacement.

```bash
# Default: Replace /usr/local/Ascend with /usr/local/npu
python3 scripts/replace_ascend_path.py
```

After execution, the script outputs the number of modified files and the total number of replacements.

### Step 4: Verify the Replacement Results

```bash
# Check if there are any unreplaced paths (the result should be 0)
grep -r "/usr/local/Ascend" . \
  --include='*.sh' --include='*.md' --include='*.rst' --include='*.py' \
  --exclude-dir='.git' | wc -l
```

## Post-Execution Verification

### 1. Environment Variable Loading Verification

```bash
# Verify that the set_env.sh file exists under the new path
ls /usr/local/npu/ascend-toolkit/set_env.sh

# Load environment variables
# Modify the ascend-toolkit path based on the actual situation
source /usr/local/npu/ascend-toolkit/set_env.sh

# Verify that the environment variables have taken effect
echo $ASCEND_HOME_PATH
```

### 2. Component Installation Verification

```bash
# Install MindSpeed MM
pip install -e .

# Verify installation
python3 -c "import mindspeed_mm; print('MindSpeed MM installed successfully')"

# Verify NPU availability
python3 -c "import torch_npu; print('NPU available:', torch_npu.npu.is_available())"
```

### 3. Core Functionality Smoke Test

Refer to the corresponding model's README to verify that the training process can start normally.

```bash
source /usr/local/npu/ascend-toolkit/set_env.sh

# Run the example script (subject to the specific model)
bash examples/<model_name>/pretrain_<model_name>.sh
```

## Script Option Description

```bash
usage: replace_ascend_path.py [-h] [--source SOURCE] [--target TARGET]
                               [--dir DIR] [--extensions EXT [EXT ...]]
                               [--dry-run]

Options:
  -h, --help            Help information
  --source SOURCE       Source path (default:/usr/local/Ascend)
  --target TARGET       Target path (default: /usr/local/npu)
  --dir DIR             Directory to be scanned (default: current director)
  --extensions EXT...   Filename extension whitelist (default: .sh .md .rst .py)
  --dry-run             Change preview; no modifications to files
```
