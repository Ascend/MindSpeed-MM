# Tool Configuration

## Applicable Backends

This document applies to the Mcore and Mcore-FSDP2 backends.

The tools include the performance profiler, the memory snapshot, and the SoRA model feature extraction tool, which are described below:

Configuration file path: `mindspeed_mm/tools/tools.json`

## Performance Profiler

### General Settings

```json
{
    "profile": {
        "enable": false,                     // Whether to enable profile collection
        "profile_type": "static",            // Dynamic or static profile collection type: static / dynamic
        "ranks": [0, 1, 2, 3],               // Ranks for profile collection; -1 collects all ranks
    },
    ......
}
```

### Static Profiling

This function collects data within a specified step range during model training. The procedure is as follows:

In the model config settings, turn on the enabling switch, set `profile_type` to `static`, and set ranks.

Configure the parameters related to static profiling:

```json
{
    "profile": {
        ......
        "static_param": {
            "level": "level1",                            // Profiling collection level: level0, level1, level2
            "with_stack": false,                          // Whether to collect the operator call stack
            "with_memory": false,                         // Whether to collect memory usage
            "record_shapes": false,                       // Whether to collect the InputShapes and InputTypes of operators
            "with_cpu": true,                             // Whether to collect CPU information
            "save_path": "./npu_profiling",               // Save path of profiling data
            "start_step": 10,                             // Step at which collection starts
            "end_step": 11,                               // Step at which collection ends
            "data_simplification": false,                 // Whether to use simplified data
            "aic_metrics_type": "PipeUtilization",        // Collection mode. PipeUtilization and ArithmeticUtilization are supported, and PipeUtilization is used by default
            "analyse_flag": true                          // Whether to enable online analysis
        },
        ......
    },
    ......
}
```

When `analyse_flag` is `false`, the generated profiling files are stored under `save_path` and must be parsed offline with the following command:

```python
python mindspeed_mm/tools/profiler.py
```

The offline parsing command supports the following parameters:

```bash
--mm-tool <path>               # Path to the MM tool configuration file; defaults to ./mindspeed_mm/tools/tools.json
--profiler-path <path>         # Path to the Profiler data directory (optional; read from the configuration file if not specified)
--max-process-number <number>  # Maximum number of processes for analysis (optional; default: number of CPU cores/2)
--export-type <type>           # Export type of the analysis result; text and db are supported and can be specified multiple times; default: text
```

**Example:**

```bash
# Parse offline using the path in the configuration file
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json

# Specify the Profiler data directory path
python mindspeed_mm/tools/profiler.py --profiler-path ./npu_profiling

# Export analysis results in multiple formats (specify the --export-type parameter multiple times)
python mindspeed_mm/tools/profiler.py --mm-tool mindspeed_mm/tools/tools.json --export-type text --export-type db
```

:::

This command parses all profiling data in the specified path. See [Offline Parsing](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/Profiling/atlasprofiling_16_0034.html).

For scenarios with large profiling files, such as long sequences, very large models, and reinforcement learning, offline parsing saves resources during training.

### Dynamic Profiling

This function can start the collection process at any time during model training. The procedure is as follows:

In the model config settings, turn on the enabling switch, set `profile_type` to `dynamic`, and set ranks.

Configure the parameters related to dynamic profiling:

```json
{
    "profile": {
        ......
        "dynamic_param": {
            "config_path": "path to dynamic config folder"       // Path to the config and log files
        }
    },
    ......
}
```

When `config_path` points to an empty folder, a `profiler_config.json` file is generated automatically.
During model execution, you can modify the `profiler_config.json` configuration at any time, and profile collection takes effect and starts from the next step.
For the `profiler_config.json` configuration, see [Collecting and Parsing Profile Data (dynamic_profile)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0033.html).

## Memory Snapshot

```json
{
    ......
    "memory_profile": {
        "enable": false,                           // Whether to enable memory collection
        "start_step": 0,                           // Step at which recording starts. The start point of training steps; 0 represents the initialization process
        "end_step": 2,                             // Step at which recording ends. The value is the start point of training steps; 0 represents the initialization process
        "save_path": "./memory_snapshot",          // Save path of the snapshot files
        "dump_ranks": [0],                         // List of ranks for snapshot recording, starting from 0
        "stacks": "all",                           // Stack information recording. Options: python/all
        "max_entries": null,                       // Maximum number of records; null means no limit
        "mem_info":  false
    },
    ......
}
```

## SoRA Model Feature Extraction

```json
{
    ......
    "sorafeature":{
        "save_path": "./sora_features",                    // Save path
        "extract_video_feature": false,                    // Whether to extract video features
        "extract_text_feature": false                       // Whether to extract text features
    }
}
```
