# Weight Conversion

## Background and Challenges

The training weight format of MindSpeed-MM depends on the training backend used, and both differ from the Hugging Face weight format, so conversion is required before use.

- **MCORE (Megatron) backend**: The weights are in the Megatron format. Taking the Qwen model as an example, in the Megatron format, the `qkv` weights in the attention mechanism and the `gate` and `up` weights in the MLP are merged together, whereas in the Hugging Face weight format they are separate. Megatron training also requires PP and TP sharding, among other things.
- **FSDP2 backend**: The weights are in the PyTorch Distributed Checkpoint (DCP) format, which is used for both loading and saving during training. Conversion to and from the Hugging Face format is required before training with `meta ini` and when exporting after training.

## Solution

`mm-convert`, a unified weight conversion tool, is developed based on the `jsonargparse` command line together with `pydantic` parameter validation, enabling mutual conversion between the weight formats of each MindSpeed-MM backend and the Hugging Face weight format.

The tool entry point is defined in `pyproject.toml`. After installing MindSpeed-MM according to the README, you can directly run the `mm-convert` command in the current environment:

```toml
[project.scripts]
mm-convert = "checkpoint.convert_cli:main"
```

Advantages:

- **Architecture design**: Weight conversion adopts an operator-based design, which greatly reduces redundant code and decouples the conversion entry point from the business logic, making the overall architecture clearer and easier to maintain.
  
- **Type safety and parameter validation**: It provides comprehensive type annotations and strong static checking capabilities, separates parameter validation from business code to ensure type safety, and supports passing parameters through shell scripts, environment variables, configuration files, and other methods, making it convenient and flexible to use.
  
- **High quality and high reliability**: It defines clear weight conversion rules and includes complete UT cases, effectively ensuring the correctness and stability of the conversion process and improving overall code quality.

## Instructions

`mm-convert` is the unified entry point shared by the two training backends. Different backends use different converters and subcommands.

| Training Backend | Converter | Common Subcommands | Weight Format |
|---|---|---|---|
| MCORE (Megatron) | Model-specific converters, such as `Qwen2VLConverter` and `InternVLConverter` | `hf_to_mm`/`mm_to_hf`/`resplit` | Megatron format (supports PP/TP sharding) |
| FSDP2 | `GenericDCPConverter` (generic, model structure-agnostic) | `hf_to_dcp`/`dcp_to_hf` | DCP format |

The converter list is continuously expanding. For the actually supported converters, refer to the output of `mm-convert -h`.

### Usage

Use the `-h` or `help` parameter to view all parameters (the following output is an example; refer to the actual output of the current version).

```bash
[root@node-xx-xx MindSpeed-MM]# mm-convert  -h
usage: mm-convert [-h] [--config CONFIG] [--print_config[=flags]] {Qwen2VLConverter,InternVLConverter} ...

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customize the output and are one or more
                        keywords separated by comma. The supported flags are: comments, skip_default, skip_null.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    Qwen2VLConverter    Qwen2VL model converter
    InternVLConverter   InternVL2.5/InternVL3 model converter
```

The supported subcommands include `Qwen2VLConverter`, `InternVLConverter`, and others.

Continuing with `help` on the `Qwen2VLConverter` command shows:

```bash
[root@node-xx-xx MindSpeed-MM]# mm-convert  Qwen2VLConverter -h
usage: mm-convert [options] Qwen2VLConverter [-h] [--config CONFIG] [--print_config[=flags]] {hf_to_mm,mm_to_hf,resplit} ...
...
subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    hf_to_mm            Convert HuggingFace model weights to MindSpeed-MM format
    mm_to_hf            Convert MindSpeed-MM model weights to HuggingFace format
    resplit             Reshard MindSpeed-MM model weights
```

Continuing with `help` on the `hf_to_mm` subcommand shows the specific parameters.

```bash
mm-convert  Qwen2VLConverter hf_to_mm -h
```

### Parameter Passing Methods (Using Qwen2VLConverter with the MCORE Training Backend as an Example)

Three parameter passing methods are supported.

#### Command-line

```bash
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "mm/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "hf/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 1
```

#### YAML File

1. The yaml configuration file can automatically generate a basic version, after which you can manually modify the necessary configuration.

   ```bash
   mm-convert  Qwen2VLConverter hf_to_mm --print_config=comments > hf_to_mm.yaml
   ```

   The following configuration file will be generated:

   ```yaml
   # HuggingFace to MindSpeed-MM weight conversion

   # Configuration for HuggingFace → MindSpeed-MM weight conversion
   cfg:

     # Path to save MindSpeed-MM weights (required, type: <class 'Path'>)
     mm_dir:

     # Parallel configuration
     parallel_config:

       # Number of pipeline parallel layers split per device in the llm module (required, type: list[list[Annotated[int, Ge(ge=0)]]])
       llm_pp_layers:

       # Number of pipeline parallel layers split per device in the vit module (required, type: list[list[Annotated[int, Ge(ge=0)]]])
       vit_pp_layers:

       # Tensor parallel group. During model conversion, different TP groups must be split into different directories (type: Annotated[int, Gt(gt=0)], default: 1)
       tp_size: 1

     # Configuration for the original weight path downloaded from HF
     hf_config:

       # Path downloaded from Hugging Face (required, type: Annotated[Path, PathType(path_type='dir')])
       hf_dir:
   ```

2. Modify the content of the configuration file.

   ```yaml
   # HuggingFace to MindSpeed-MM weight conversion

   #  Configuration for HuggingFace → MindSpeed-MM weight conversion
   cfg:

     # Path to save MindSpeed MM weights (required, type: <class 'Path'>)
     mm_dir: "mm/Qwen2-VL-7B-Instruct"

     # Parallel configuration
     parallel_config:

       # Number of pipeline parallel layers split per device in the llm module (required, type: list[Annotated[int, Ge(ge=0)]])
       llm_pp_layers: [[1,10,10,7]]

       # Number of pipeline parallel layers split per device in the vit module (required, type: list[Annotated[int, Ge(ge=0)]])
       vit_pp_layers: [[32,0,0,0]]

       # Tensor parallel group. During model conversion, different TP groups must be split into different directories (type: Annotated[int, Gt(gt=0)], default: 1)
       tp_size: 1

     # Configuration of the original weight path downloaded from HF
     hf_config:

       # Path downloaded from Hugging Face (required, type: Annotated[Path, PathType(path_type='dir')])
       hf_dir: "hf/Qwen2-VL-7B-Instruct"
   ```

3. Run the conversion script.

   ```bash
   mm-convert  Qwen2VLConverter hf_to_mm --config hf_to_mm.yaml
   ```

#### Environment Variables

1. The environment variable mode is disabled by default. To enable it, configure as follows.

   ```bash
   export JSONARGPARSE_DEFAULT_ENV=true
   ```

2. Use the `-h` command to view the environment variables corresponding to each configuration.

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

4. Execute the following command.

   ```bash
   mm-convert  Qwen2VLConverter hf_to_mm
   ```

### FSDP2 Backend Weight Conversion (DCP Format)

The FSDP2 backend uses the generic converter `GenericDCPConverter`, which does not distinguish specific model structures. The subcommands are `hf_to_dcp` and `dcp_to_hf`.

Convert Hugging Face weights to DCP (before training, used with `meta ini`) format:

```bash
mm-convert GenericDCPConverter hf_to_dcp \
  --hf_dir ckpt/Qwen3-VL-30B-Instruct \
  --dcp_dir ckpt/Qwen3-VL-30B-Instruct-dcp
```

After conversion, a `release/` directory and `latest_checkpointed_iteration.txt` are generated under `--dcp_dir`.

Convert DCP weights back to Hugging Face format (after training):

```bash
mm-convert GenericDCPConverter dcp_to_hf \
  --load_dir save_dir/release \
  --save_dir save_dir_hf \
  --model_assets_dir ckpt/Qwen3-VL-30B-Instruct
```

Where:

- `--load_dir`: the directory that stores the DCP shards (the `release` directory under the training save path)
- `--save_dir`: the output directory for the exported Hugging Face weights
- `--model_assets_dir`: the original Hugging Face weight directory, used to copy assets such as `config` and `tokenizer`

For the complete set of parameters (such as `dcp_prefix`, `hf_prefix`, `tie_weight_mapping`, `fused_linear_names`, etc.), refer to `mm-convert GenericDCPConverter hf_to_dcp -h` and `mm-convert GenericDCPConverter dcp_to_hf -h`.
