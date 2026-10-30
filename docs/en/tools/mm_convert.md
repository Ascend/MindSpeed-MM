
# Weight Conversion Tool Usage
 
The new weight conversion interface was introduced on 2025/1/20 and currently supports conversion of various Qwen2VL model sizes.

The weight conversion interface defines the following command-line arguments in `pyproject.toml`. After `mindspeed-mm` is installed according to the `README`, the `mm-convert` tool can be executed in the current environment.

```toml
[project.scripts]
mm-convert = "checkpoint.convert_cli:main"
```

How to Use:

```bash
[root@node-xx-xx MindSpeed-MM]# mm-convert  -h
usage: mm-convert [-h] [--config CONFIG] [--print_config[=flags]] {Qwen2VLConverter,InternVLConverter} ...

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more
                        keywords separated by comma. The supported flags are: comments, skip_default, skip_null.

Subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    Qwen2VLConverter    Qwen2VL model conversion tool
    InternVLConverter   InternVL2.5/InternVL3 model conversion tool

```

The available subcommands include `Qwen2VLConverter` and `InternVLConverter`.

Running the `help` command for Qwen2VLConverter shows:

```bash
[root@node-xx-xx MindSpeed-MM]# mm-convert  Qwen2VLConverter -h
usage: mm-convert [options] Qwen2VLConverter [-h] [--config CONFIG] [--print_config[=flags]] {hf_to_mm,mm_to_hf,resplit} ...
...
subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    hf_to_mm            Convert huggingface model weights to mindspeed-mm model weights
    mm_to_hf            Convert mindspeed-mm model weights to huggingface model weights
    resplit             Re-shard mindspeed-mm model weights

```

Continue to run `help` on the subcommand `hf_to_mm` to view the specific parameters.

```bash
mm-convert  Qwen2VLConverter hf_to_mm -h
```

# Parameter Passing Methods

Three parameter passing methods are supported.

## Command-Line Arguments

```bash
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "mm/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "hf/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 1
```

> The parallelism configuration used for conversion (`tp_size`, `llm_pp_layers`/`vit_pp_layers`) must be consistent with the training script; otherwise, the generated weight directory names and sharding will not match, which causes loading failures.
> For example, the weight directory only includes the stage suffix (e.g., `mp_rank_00_000`) when PP > 1. When PP = 1, the directory is `mp_rank_00`, and `llm_pp_layers` should be configured as a single segment (e.g., [[28]]).

## YAML File

1. A basic version of the YAML configuration file can be generated automatically, and then the necessary configurations can be modified manually.

     ```bash
      mm-convert  Qwen2VLConverter hf_to_mm --print_config=comments > hf_to_mm.yaml
     ```

     The following configuration file is generated.

     ```yaml
     # Convert Hugging Face weights to MindSpeed MM weights
     
     # Configuration for converting Hugging Face weights to Mindspeed MM weights
     cfg:
     
       # Path to save the MM weights (required, type: <class 'Path'>)
       mm_dir:
     
       # Parallelism configuration
       parallel_config:
     
         # Number of layers on each device for the pipeline parallel split of the llm module (required, type: list[list[Annotated[int, Ge(ge=0)]]])
         llm_pp_layers:
     
         # Number of layers on each device for the pipeline parallel split of the vit module (required, type: list[list[Annotated[int, Ge(ge=0)]]])
         vit_pp_layers:
     
         # Tensor parallel group; during model conversion, different tp groups must be sharded into different directories (type: Annotated[int, Gt(gt=0)], default: 1)
         tp_size: 1
     
       # Configuration of the path to the original weights downloaded from hf
       hf_config:
     
         # Path of the Hugging Face download (required, type: Annotated[Path, PathType(path_type='dir')])
         hf_dir:
     ```

2. Modify the content of the configuration file.

     ```yaml
     # Convert Hugging Face weights to MindSpeed MM weights
     
     # Configuration for converting Hugging Face weights to MindSpeed MM weights
     cfg:
     
       # Path to save the MM weights (required, type: <class 'Path'>)
       mm_dir: "mm/Qwen2-VL-7B-Instruct"
     
       # Parallelism configuration
       parallel_config:
     
         # Number of layers on each device for the pipeline parallel split of the llm module (required, type: list[Annotated[int, Ge(ge=0)]])
         llm_pp_layers: [[1,10,10,7]]
     
         # Number of layers on each device for the pipeline parallel split of the vit module (required, type: list[Annotated[int, Ge(ge=0)]])
         vit_pp_layers: [[32,0,0,0]] 
     
         # Tensor parallel group; during model conversion, different tp groups must be sharded into different directories (type: Annotated[int, Gt(gt=0)], default: 1)
         tp_size: 1
     
       # Configuration of the path to the original weights downloaded from hf
       hf_config:
     
         # Path of the Hugging Face download (required, type: Annotated[Path, PathType(path_type='dir')])
         hf_dir: "hf/Qwen2-VL-7B-Instruct"
     ```

3. Run the conversion script.

     ```bash
     mm-convert  Qwen2VLConverter hf_to_mm --config hf_to_mm.yaml
     ```

## Environment Variables

1. The environment variable mode is disabled by default. To enable it, configure as follows.

     ```bash
      export JSONARGPARSE_DEFAULT_ENV=true
     ```

2. Use the `-h` command to view the environment variable corresponding to each configuration.

     ```bash
     mm-convert  Qwen2VLConverter hf_to_mm -h
     ```

3. Configure the environment variables.

     ```bash
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__PARALLEL_CONFIG__LLM_PP_LAYERS="[[1,10,10,7]]"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__PARALLEL_CONFIG__VIT_PP_LAYERS="[[32,0,0,0]]"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__PARALLEL_CONFIG__TP_SIZE="1"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__HF_CONFIG__HF_DIR="hf/Qwen2-VL-7B-Instruct"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__MM_DIR="mm/Qwen2-VL-7B-Instruct"
     ```

4. Run the conversion command.

     ```bash
     mm-convert  Qwen2VLConverter hf_to_mm
     ```

# Notes on Weight Conversion

If an error occurs during weight conversion, please download the latest weights from websites such as Hugging Face and reconvert them. If the issue persists, please submit an issue, and we will address it promptly.

> **Why is it necessary to download the latest weights?**
> 
> During the weight conversion process, related transformers APIs (such as `AutoConfig`) are called. With version upgrades of transformers, old version weight files may become incompatible with newer versions of transformers. Therefore, it is recommended to use the latest weights to ensure successful conversion.
