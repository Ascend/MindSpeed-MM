# MindSpeed MM Toolkit Usage

## Mcore-FSDP2 Backend Tools

### Profiling Tool

The suite integrates the Ascend [profiling tool](../../../mindspeed_mm/tools/profiler.py) to provide analysis of model execution. Built-in models are already adapted. You only need to modify the [tools.json](../../../mindspeed_mm/tools/tools.json) file to enable it.

+ If adding a new model, please first adapt the following settings:

```python
from mindspeed_mm.tools.profiler import Profiler

prof = Profiler(prof_config)
prof.start()
while train():
    train_one_step
    prof.step()
prof.stop()
```

+ General model `config` settings:

```bash
--enable                  # Enable profiling
--profile_type            # Profiling type: static or dynamic
--ranks                   # Ranks to profile. The default value is -1, meaning all ranks are profiled.
```

#### Static Profiling

This function collects data for a specified range of steps during model training. The steps are as follows:

1. In the model config settings, enable profiling via `enable`, set `profile_type` to `static`, and set `ranks`.

2. Configure the relevant parameters for static profiling.

    The specific settings for `static_param` of static profiling are as follows:

    ```bash
    --level                     # Profiling level: level0, level1, level2
    --with_stack                # Whether to collect the operator call stack
    --with_memory               # Whether to collect memory usage
    --record_shapes             # Whether to collect the InputShapes and InputTypes of operators
    --with_cpu                  # Whether to collect CPU information
    --save_path                 # Save path of profiling files
    --start_step                # Step at which to start profiling
    --end_step                  # Step at which to end profiling
    --data_simplification       # Whether to use simplified data
    --aic_metrics_type          # Profiling mode: PipeUtilization and ArithmeticUtilization are supported. PipeUtilization is used by default
    --analyse_flag              # Whether to enable online parsing
    ```

3. Run the model and collect profiling file.

4. Parse the profiling file.

    + When `analyse_flag` is `true`, parsing will be automatically executed during model execution.

    + When `analyse_flag` is `false`, the generated profiling file is located in the `save_path` path, and the following command is required to trigger offline parsing:

        ```shell
        python mindspeed_mm/tools/profiler.py
        ```

#### Dynamic Profiling

This function allows you to start profiling at any time during model training. The steps are as follows:

1. In the model `config` settings, enable profiling via `enable`, set `profile_type` to `dynamic`, and configure `ranks`.

2. Configure the relevant parameters for dynamic profiling.

    The specific settings for `dynamic_param` of dynamic profiling are as follows:

    ```bash
    --config_path               # Path of the config and log files
    ```

    - `config_path` points to an empty folder and automatically generates the `profiler_config.json` file.
    - `config_path` refers to the path of an existing dynamic configuration file `profiler_config.json`.

3. Run the model.

4. During model execution, modify `profiler_config.json` configuration at any time, and profiling will take effect and start at the next step:
    - The dynamic profiling function determines whether the file has been modified by recognizing the status of the `profiler_config.json` file. If a modification to the `profiler_config.json` file is detected, `dynamic_profile` will start the profiling task at the next step.
    - The diagnostic log for `dynamic_profile` is automatically recorded in the `config_path` directory.

For the specific parameters, input parameter table, and operation steps of dynamic profiling, refer to [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0033.html) in the *CANN Performance Tuning Tool User Guide*.

#### Offline Parsing Command Options

```shell
--mm-tool <path>               # MindSpeed MM tool configuration file path; defaults to ./mindspeed_mm/tools/tools.json
--profiler-path <path>         # Profiler data path (optional; read from the configuration file if not specified)
--max-process-number <number>  # Maximum number of processes to analyze (optional; default: number of CPU cores/2)
--export-type <type>           # Export type for analysis results. Supported types: text, db. Can be specified multiple times. Default: text
```

|  Type | Option | Description | Default Value |
|-|-|-|-|
| path | `--mm-tool` | Path to the Mindspeed MM tool configuration file | `./mindspeed_mm/tools/tools.json` |
| path | `--profiler-path` | Path to the Profiler data | Read from the configuration file. |
| number | `--max-process-number` | Maximum number of processes for analysis | Number of CPU cores/2 |
| type | `--export-type` | Export type for analysis results. Supported types include `text` and `db`. Can be specified multiple times. | `text` |

**Example**

```shell
# Perform offline parsing using the path in the configuration file
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json

# Specify the Profiler data path
python mindspeed_mm/tools/profiler.py --profiler-path ./npu_profiling

# Export analysis results in multiple formats (specify the --export-type parameter multiple times)
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json --export-type text --export-type db
```

This command parses all profiling data under the specified path. For details, refer to the "[Offline Parsing](https://www.hiascend.com/document/detail/en/canncommercial/900/devaids/Profiling/atlasprofiling_16_0034.html)" section in the *CANN Performance Tuning Tool User Guide*.

For scenarios with large profiling files, such as long sequences, extremely large models, and reinforcement learning, offline parsing reduces resource usage during training.

### Sora-like Model Feature Extraction

The tools in the [feature_extraction](../../../mindspeed_mm/tools/feature_extraction) directory can extract and save video and text features. Currently, feature extraction is supported for single-batch static datasets.

1. Modify the video feature, text feature, and data storage path settings in the [tools.json](../../../mindspeed_mm/tools/tools.json) file as needed.

    ```bash
    --extract_video_feature # Whether to extract video features
    --extract_text_feature  # Whether to extract text features
    --save_path             # Feature data storage path
    ```

2. Before use, modify the dataset and configuration file (VAE, T5) paths of the corresponding model in the [feature_extraction.sh](../../../examples/wan2.1/feature_extract/feature_extraction.sh) file as needed.

    ```bash
    --MM_DATA       # Data configuration file path (.json)
    --MM_MODEL      # Model configuration file path (.json)
    ```

3. After configuration, call [feature_extraction.sh](../../../examples/wan2.1/feature_extract/feature_extraction.sh) to extract data features.

### Memory Snapshot Extraction

The suite integrates the Ascend [memory snapshot collection tool](../../../mindspeed_mm/tools/mem_profiler.py) to provide analysis of model execution. Built-in models are already adapted; you only need to modify the [tools.json](../../../mindspeed_mm/tools/tools.json) file to enable it.

+ For models that reuse the [training pipeline](../../../mindspeed_mm/training.py), you only need to modify the configuration. The supported configuration items are as follows.

```json5
{
  "memory_profile": {
    "enable": false,    // Switch for memory profiling
    "start_step": 0,    // Step at which to start recording. The value is the starting point of the training steps; 0 represents the step at which recording starts
    "end_step": 2,      // Step at which to end recording. The value is the ending point of the training steps; 2 represents the step at which recording ends
    "save_path": "./memory_snapshot",  // Snapshot file save path
    "dump_ranks": [     // List of ranks to record snapshots, starting from 0
      0
    ],
    "stacks": "all",    // Stack information recording. Options: python/all
    "max_entries": null, // Maximum number of records; unlimited if null
    "mem_info": false    // Whether to record memory information
  }
}
```

+ For an independent training process, refer to the following code to adapt the training script to use the profiler feature. The parameter configuration is the same as above.

```python
from megatron.training import get_args
from mindspeed_mm.tools.mem_profiler import memory_profiler

args = get_args()                                   # Get configuration
memory_profiler.reset(args.mm.tool.memory_profile)  # Refresh the profiler state with the configuration
training_preparation()                              # Run the training preparation code
while iteration < args.train_iters:                 # Main training loop
    memory_profiler.step()                          # Call the profiler to record one iteration
    train_one_step()                                # Train one iteration
memory_profiler.stop()                              # Stop profiling
```

+ For scripts that do not have a typical training structure, or for partial manual debugging, you can directly call the base functions and modify them according to your requirements.

```python
code_not_record()
from mindspeed_mm.tools.mem_profiler import _record
_record()
code_to_record()
```

+ The dump and recording start operations can be in different files.

```python
code_to_record()
from mindspeed_mm.tools.mem_profiler import _dump, _stop
_dump()
_stop()
```

After the dump is complete, a `pickle` file starting with `snapshot_` will be generated in the output directory. You can visualize the memory snapshot in an [interactive viewer](https://pytorch.org/memory_viz).

### TensorBoard Usage

1. Install TensorBoard.

    ```bash
    pip install tensorboard
    ```

2. Set the TensorBoard save path `TENSORBOARD_LOGS_PATH`, then add `--tensorboard-dir` to `OUTPUT_ARGS` in the running script to enable it.

    ```shell
    TENSORBOARD_LOGS_PATH="./tensorboard_dir/" # TensorBoard save path

    # Add --tensorboard-dir to args
    OUTPUT_ARGS="
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --load $LOAD_PATH \
        --save $SAVE_PATH \
        --ckpt-format torch \
        --tensorboard-dir $TENSORBOARD_LOGS_PATH \
    "
    ```

3. Open TensorBoard for result viewing. `./tensorboard_dir/` is the save path from step 2. Modify it as needed.

    ```shell
    tensorboard --logdir ./tensorboard_dir/
    ```

## FSDP2 Backend Tools

### Static Profiling

The FSDP2 backend profiles data by configuring the `tools`-related parameters in the model training YAML file.

1. Configure `tools.profile` in the model training YAML file, for example:

    ```yaml
    tools:
      profile:
        enable: true
        profile_type: static
        ranks: [0]
        static_param:
          level: level1
          with_stack: false
          with_memory: false
          record_shapes: false
          with_cpu: true
          save_path: ./profiling
          start_step: 10
          end_step: 11
          data_simplification: false
          aic_metrics_type: PipeUtilization
          analyse_flag: true
    ```

    The main parameters are described as follows:

    ```bash
    --enable                  # Whether to enable profiling
    --profile_type            # The FSDP2 backend currently supports static only
    --ranks                   # List of ranks to collect; [-1] means all ranks
    --level                   # Profiling level: level0, level1, level2
    --with_stack              # Whether to collect the operator call stack
    --with_memory             # Whether to collect operator memory usage
    --record_shapes           # Whether to collect the InputShapes and InputTypes of operators
    --with_cpu                # Whether to collect CPU information
    --save_path               # Save path of profiling files
    --start_step              # Training step at which to start profiling
    --end_step                # Training step at which to end profiling, excluded
    --data_simplification     # Whether to use simplified data
    --aic_metrics_type        # PipeUtilization and ArithmeticUtilization are supported
    --analyse_flag            # Whether to enable online parsing
    ```

2. Start training as described in the model document. The training entry automatically reads `tools.profile` and starts, advances, and stops profiling in the training loop. No model code changes are required.

3. When `start_step` is set to `0`, the profiler starts before components such as the model, optimizer, and DataLoader are initialized. Therefore, the results also cover the initialization before training and continue to be collected for subsequent training iterations until `end_step` is reached.

4. Profiling files are saved in the directory specified by `save_path`. When `analyse_flag` is `true`, parsing is performed automatically during training. When it is `false`, use the FSDP2 offline parsing command described below.

### Offline Profiling Parsing

The FSDP2 backend uses `mindspeed_mm/fsdp/tools/profiler.p`y for offline parsing and `--profiler-path` to specify the profiling data directory to be parsed.

```shell
--profiler-path <path>         # Profiler data path (required)
--max-process-number <number>  # Maximum number of processes to analyze (optional; default: number of CPU cores/2)
--export-type <type>           # Export type. Supported types: text, db. Can be specified multiple times. Default: text
```

|Parameter Type|Parameter Name|Description|Default Value|
|-|-|-|-|
|path|--profiler-path|Profiler data path|None; must be specified|
|number|--max-process-number|Maximum number of processes to analyze|Number of CPU cores/2|
|type|--export-type|Export type for analysis results. Supported types: text, db. Can be specified multiple times.|text|

**Example**

```shell
# Parse the profiling data in the specified directory
python mindspeed_mm/fsdp/tools/profiler.py --profiler-path ./profiling

# Limit the maximum number of processes used for offline parsing
python mindspeed_mm/fsdp/tools/profiler.py \
    --profiler-path ./profiling \
    --max-process-number 8

# Export both text and db formats
python mindspeed_mm/fsdp/tools/profiler.py \
    --profiler-path ./profiling \
    --export-type text \
    --export-type db
```

### Memory Snapshot Collection

The FSDP2 backend uses a standalone [memory snapshot collection tool](../../../mindspeed_mm/fsdp/tools/memory_profiler.py), which is configured through `tools.memory_profile` in the model training YAML file.

1. Configure `tools.memory_profile` in the model training YAML file as follows.

    ```yaml
    tools:
      memory_profile:
        enable: true
        start_step: 1
        end_step: 2
        save_path: ./memory_snapshot
        dump_ranks: [0]
        stacks: all
        max_entries: null
        mem_info: false
    ```

    The parameters are described as follows:

    ```bash
    --enable                  # Whether to enable memory snapshot collection
    --start_step              # Step at which to start recording memory history
    --end_step                # Step at which to export the memory snapshot and stop recording
    --save_path               # Memory snapshot save path
    --dump_ranks              # List of ranks to export memory snapshots, starting from 0
    --stacks                  # Stack information scope. Options: python or all
    --max_entries             # Maximum number of records; null means unlimited
    --mem_info                # Whether to print memory usage at each step
    ```

2. Start training as described in the model document. The training entry automatically reads `tools.memory_profile` and starts recording, exports snapshots, and stops recording in the training loop. No model code changes are required.

3. After `end_step` is reached, `pickle` files starting with `snapshot_` and distinguished by rank are generated in the `save_path` directory. You can analyze them with the [interactive viewer](https://pytorch.org/memory_viz). If training ends before `end_step` is reached, the memory history already recorded is exported when training exits.
