# Evaluation and Validation During Training

## Problem Analysis

During long training runs, observing only the training set loss makes it hard to judge in time how the model performs on unseen data, so overfitting may occur without being noticed. To observe the validation set loss periodically during training, the training pipeline needs to support validation set configuration, validation frequency control, and model state switching during the validation phase.

## Solution

MindSpeed MM supports periodic validation during training. When training reaches the specified interval, the framework pauses the training branch, switches the model to the `eval()` state, iterates over the validation set to compute the validation loss, and restores the training state to continue training after validation finishes.

## How to Use

FSDP2 adds the in-training validation capability in this release, while the Mcore (Megatron) training pipeline already has validation. This document describes how to use both backends. Their configuration entries differ: FSDP2 uses the new fields `val_dataset_param` and `val_interval`, while Mcore uses Megatron's original `eval_interval` and `eval_iters` as well as the validation set fields in the dataset.

### Native FSDP2

Add the validation set configuration to the model YAML configuration file and set the validation interval:

```yaml
data:
  dataset_param: &TRAIN_DATASET_PARAM
    dataset_type: huggingface
    basic_parameters: &TRAIN_BASIC_PARAMETERS
      dataset_dir: ./data
      dataset: ./data/train.json
      cache_dir: ./cache_dir/train

  val_dataset_param:
    <<: *TRAIN_DATASET_PARAM
    basic_parameters:
      <<: *TRAIN_BASIC_PARAMETERS
      dataset: ./data/val.json
      cache_dir: ./cache_dir/val

training:
  micro_batch_size: 4
  val_interval: 1000
  val_micro_batch_size: 1
```

- `data.val_dataset_param`: validation set configuration, with the same structure as `data.dataset_param`. YAML anchors can be used to reuse the training set configuration and override only the fields that differ, such as the validation set path and the cache path.
- `training.val_interval`: validation interval in steps. The default is `0`, which disables validation. A value greater than `0` enables validation.
- `training.val_micro_batch_size`: micro batch size for the validation phase. If it is not set, `training.micro_batch_size` is reused by default.

Once validation is enabled, an error is reported at training startup if no validation set is configured. The validation dataloader disables shuffle and uses an independent seed, so that the sampling order of the training dataloader is not affected.

See `examples/qwen3_5/qwen3_5_35B_config.yaml` for reference.

For FSDP2, some models can refer to the implementation of Qwen3.5. The support status of other models depends on their specific implementations.

### Mcore (Megatron)

The Mcore training pipeline uses Megatron's original evaluation parameters to control the validation frequency and the number of validation steps. When launched through a shell script, the related parameters are usually set in `OUTPUT_ARGS`:

```shell
OUTPUT_ARGS="
    ...
    --eval-interval 1000 \
    --eval-iters 10 \
    ...
"
```

- `--eval-interval`: validation interval in iterations.
- `--eval-iters`: number of iterations executed for each validation.

The validation set can be configured explicitly through `val_dataset` in `basic_parameters` of the data configuration:

```json
{
  "dataset_param": {
    "basic_parameters": {
      "dataset": "./data/train.json",
      "val_dataset": "./data/val.json",
      "val_max_samples": null
    }
  }
}
```

- `val_dataset`: explicitly specifies the validation set path.
- `val_max_samples`: limits the number of validation set samples; commonly used for debugging.

On the Mcore (Megatron) side, the validation capability depends on the implementation of each model training pipeline. Some models can refer to the implementation of Qwen2.5VL. The support status of other models depends on their specific implementations.

### Constraints

- This feature currently supports only loss validation during training; the model forward must return the loss, and the batch format of the validation set must be the same as that of the training set.
- Generative evaluation and task metric evaluation are currently not supported.
