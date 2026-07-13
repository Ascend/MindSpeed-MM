# MindSpeed MM Toolkit Usage

## Profiling Tool

The suite integrates the Ascend [profiling tool](../../mindspeed_mm/tools/profiler.py) to provide analysis of model execution. Built-in models are already adapted; you only need to modify the [tools.json](../../mindspeed_mm/tools/tools.json) file to enable it.

+ If adding a new model, please first adapt the following settings:

```python
from mindspeed_mm.tools.profiler import Profiler

prof = Profiler(prof_config)
prof.start()
while train:
    train_one_step
    prof.step()
prof.stop()
```

+ General model `config` settings:

```bash
--enable                  # Enable profiling
--profile_type            # Dynamic or static profiling
--ranks                   # Ranks for profiling. The default is -1, meaning all ranks are profiled.
```

### Static Profiling

This function collects data for a specified range of steps during model training. The operation steps are as follows:

1. In the model `config` settings, enable profiling via `enable`, set `profile_type` to static, and configure `ranks`.

2. Configure the relevant parameters for static collection.

    The specific settings for `static_param` of static profiling are as follows:

    ```bash
    --level                     # Profiling level: level 0, level 1, level 2
    --with_stack                # Whether to profile the operator call stack
    --with_memory               # Whether to profile memory usage
    --record_shapes             # Whether to profile the InputShapes and InputTypes of operators
    --with_cpu                  # Whether to profile CPU information
    --save_path                 # Save path to profiling files
    --start_step                # Step to start profiling
    --end_step                  # Step at which to end profiling
    --data_simplification       # Whether to use simplified data for profiling
    --aic_metrics_type          # Profiling mode: PipeUtilization (default) and ArithmeticUtilization
    --analyse_flag              # Whether to enable online parsing
    ```

3. Run the model and collect profiling file.

4. Parse the profiling file.

    + When `analyse_flag` is `true`, parsing will be automatically executed during model execution.

    + When `analyse_flag` is `false`, the generated profiling file is located in the `save_path` path, and the following command is required to trigger offline parsing:

        ```shell
        python mindspeed_mm/tools/profiler.py
        ```

### Dynamic Profiling

This function allows you to start profiling at any time during model training. The steps are as follows:

1. In the model `config` settings, enable profiling via `enable`, set `profile_type` to static, and configure `ranks`.

2. Configure the relevant parameters for dynamic profiling.

    The specific settings for `dynamic_param` of dynamic profiling are as follows:

    ```bash
    --config_path               # Path for config and log files
    ```

    - `config_path` points to an empty folder and automatically generates the `profiler_config.json` file.
    - `config_path` refers to the path of an existing dynamic configuration file `profiler_config.json`.

3. Run the model.

4. During model execution, modify `profiler_config.json` configuration at any time, and profiling will take effect and start at the next step:
    - The dynamic profiling function determines whether the file has been modified by recognizing the status of the `profiler_config.json` file. If a modification to the `profiler_config.json` file is detected, `dynamic_profile` will start the profiling task at the next step.
    - The diagnostic log for `dynamic_profile` is automatically recorded in the `config_path` directory.

For specific parameters and detailed operation steps for dynamic profiling, please refer to the "Ascend PyTorch Profiler APIs" section in the *CANN Performance Tuning Tool User Guide*.

### Offline Parsing Command Options

```shell
--mm-tool <path>               # MindSpeed MM tool configuration file path; defaults to ./mindspeed_mm/tools/tools.json
--profiler-path <path>         # Profiler data path (optional; read from the configuration file if not specified)
--max-process-number <number>  # Maximum number of processes to analyze (optional; defaulted to number of CPU cores/2)
--export-type <type>           # Export type for analysis results. Supported types: text, db. Can be specified multiple times. Default: text
```

|  Type | Option | Description | Default Value |
|-|-|-|-|
| path | `--mm-tool` | Path to the Mindspeed MM tool configuration file | `./mindspeed_mm/tools/tools.json` |
| path | `--profiler-path` | Path to the Profiler data | Read from the configuration file. |
| number | `--max-process-number` | Maximum number of processes for analysis | Number of CPU cores/2 |
| type | `--export-type` | Export type for analysis results. Supported types include `text` and `db`. Can be specified multiple times | `text` |

**Example**:

```shell
# Perform offline parsing using the path specified in the configuration file
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json

# Specify the path to the Profiler data
python mindspeed_mm/tools/profiler.py --profiler-path ./npu_profiling

# Export analysis results in multiple formats (specify the --export-type parameter multiple times)
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json --export-type text --export-type db
```

This command parses all profiling data under the specified path. For details, refer to the "[Offline Parsing](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0034.html)" section in the *CANN Performance Tuning Tool User Guide*.

For scenarios with large profiling files, such as ultra-long sequences, extremely large models, and reinforcement learning, using offline parsing can save resource usage during training.

## Sora-like Model Feature Extraction

The tools in the [feature_extraction](../../mindspeed_mm/tools/feature_extraction) directory can be used to extract and save video and text features, currently supporting feature extraction for single-batch static datasets.

1. Modify the relevant configurations for video, text features, and data storage paths in the [tools.json](../../mindspeed_mm/tools/tools.json) file as needed.

    ```bash
    --extract_video_feature # Whether to extract video features
    --extract_text_feature  # Whether to extract text features
    --save_path             # Feature data storage path
    ```

2. Before use, modify the paths for the corresponding model dataset and configuration files (VAE, T5) in the [feature_extraction_t2v.sh](../../examples/cogvideox/feature_extract/feature_extraction_t2v.sh) file as needed.

    ```bash
    --MM_DATA       # Data configuration file path (.json)
    --MM_MODEL      # Model configuration file path (.json)
    ```

3. After configuration, call [feature_extraction_t2v.sh](../../examples/cogvideox/feature_extract/feature_extraction_t2v.sh) to extract data features.

## Memory Snapshot Extraction

The suite integrates the Ascend [memory snapshot collection tool](../../mindspeed_mm/tools/mem_profiler.py) to provide analysis of model operation. All built-in models have been adapted; simply modify the [tools.json](../../mindspeed_mm/tools/tools.json) file to enable this feature.

+ For models that reuse the [training pipeline](../../mindspeed_mm/training.py), you only need to modify the configuration. The supported configuration items are as follows.

```json5
{
  "memory_profile": {
    "enable": false,    // Memory profiling switch
    "start_step": 0,    // Step to start recording. The value is the starting point of the training steps, where 0 represents the step at which recording begins.
    "end_step": 2,      // Step to end recording. The value is the ending point of the training steps, where 2 represents the step at which recording ends.
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
memory_profiler.reset(args.mm.tool.memory_profile)  # Use configuration to refresh Profiler status
training_preparation()                              # Run training preparation code
while iteration < args.train_iters:                 # Main training loop
    memory_profiler.step()                          # Call profiler to record one iteration
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

## TensorBoard Usage

1. Install TensorBoard.

    ```bash
    pip install tensorboard
    ```

2. Set the TensorBoard save path `TENSORBOARD_LOGS_PATH`, then add `--tensorboard-dir` to `OUTPUT_ARGS` in the running script to enable it.

    ```shell
    TENSORBOARD_LOGS_PATH="./tensorboard_dir/" # TensorBoard save path

    # Add --tensorboard-dir in args
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
