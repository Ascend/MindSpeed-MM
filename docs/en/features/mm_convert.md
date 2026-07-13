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

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    Qwen2VLConverter    Qwen2VL conversion tool
    InternVLConverter   InternVL2.5/InternVL3 conversion tool

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
    hf_to_mm            Convert Hugging Face model to MindSpeed MM weights
    mm_to_hf            Convert MindSpeed MM weights to Hugging Face weights
    resplit             Re-shard MindSpeed MM weights

```

Running the `help` command for the `hf_to_mm` subcommand shows the specific parameters:

```bash
mm-convert  Qwen2VLConverter hf_to_mm -h
```

## Parameter Passing Methods

Three parameter passing methods are supported:

### Command-Line Arguments

```bash
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "mm/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "hf/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 1
```

### YAML File

A basic version of the YAML configuration file can be automatically generated, and necessary configurations can be modified manually.

     ```bash
      mm-convert  Qwen2VLConverter hf_to_mm --print_config=comments > hf_to_mm.yaml
     ```

This generates the following configuration file:

     ```yaml
     # Convert Hugging Face weights to MindSpeed MM weights

     # Configuration for converting Hugging Face weights to MindSpeed-MM weights
     cfg:

       # Path to save the MindSpeed MM weights(required, type: <class 'Path'>)
       mm_dir:

       # Parallel configuration
       parallel_config:

         # Number of layers per card for LLM module pipeline parallel partitioning (required, type: list[list[Annotated[int, Ge(ge=0)]]])
         llm_pp_layers:

         # Number of layers per card for ViT module pipeline parallel partitioning (required, type: list[list[Annotated[int, Ge(ge=0)]]])
         vit_pp_layers:

         # Tensor parallel group; during conversion, different TP groups must be sharded to different directories (type: Annotated[int, Gt(gt=0)], default: 1)
         tp_size: 1

       # Configuration for the original weights downloaded from Hugging Face
       hf_config:

         # Path to the HuggingFace download (required, type: Annotated[Path, PathType(path_type='dir')])
         hf_dir:
     ```

Modify the content in the configuration file:

     ```yaml
     # Convert Hugging Face weights to MindSpeed MM weights

     # Configuration for converting Hugging Face weights to MindSpeed MM weights
     cfg:

       # Path to save MindSpeed MM weights (required, type: <class 'Path'>)
       mm_dir: "mm/Qwen2-VL-7B-Instruct"

       # Parallel Configuration
       parallel_config:

         # Number of layers per card for LLM module pipeline parallel partitioning (required, type: list[Annotated[int, Ge(ge=0)]])
         llm_pp_layers: [[1,10,10,7]]

         # Number of layers per card for ViT module pipeline parallel partitioning (required, type: list[Annotated[int, Ge(ge=0)]])
         vit_pp_layers: [[32,0,0,0]]

         # Tensor parallel group; during conversion, different TP groups must be sharded to different directories (type: Annotated[int, Gt(gt=0)], default: 1)
         tp_size: 1

       # Configuration for original weights downloaded from Hugging Face
       hf_config:

         # Path to the HuggingFace download (required, type: Annotated[Path, PathType(path_type='dir')])
         hf_dir: "hf/Qwen2-VL-7B-Instruct"
     ```

  Execute the conversion script:

     ```bash
     mm-convert  Qwen2VLConverter hf_to_mm --config hf_to_mm.yaml
     ```

### Environment Variables

Environment variable mode is currently disabled by default. To enable it, you can configure the environment variable:

     ```bash
      export JSONARGPARSE_DEFAULT_ENV=true
     ```

Use the `-h` command to view the environment variables for each configuration:

     ```bash
     mm-convert  Qwen2VLConverter hf_to_mm -h
     ```

Configure environment variables:

     ```bash
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__PARALLEL_CONFIG__LLM_PP_LAYERS="[[1,10,10,7]]"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__PARALLEL_CONFIG__VIT_PP_LAYERS="[[32,0,0,0]]"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__PARALLEL_CONFIG__TP_SIZE="1"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__HF_CONFIG__HF_DIR="Qwen2-VL-7B-Instruct"
     export MM_CONVERT_QWEN2VLCONVERTER__HF_TO_MM__CFG__MM_DIR="hf/Qwen2-VL-7B-Instruct"
     ```

Execute:

     ```bash
     mm-convert  Qwen2VLConverter hf_to_mm
     ```

## Weight Conversion Notes

If an error occurs during weight conversion, please download the latest weights from websites such as Hugging Face and reconvert them. If the issue persists, please submit an issue, and we will address it promptly.

> **Why is it necessary to download the latest weights?**
>
> During the weight conversion process, related transformers APIs (such as `AutoConfig`) are called. With version upgrades of transformers, old version weight files may become incompatible with newer versions of transformers. Therefore, it is recommended to use the latest weights to ensure successful conversion.
