# MultiModal Dataset

## Multi-Dataset Training

### How to Use (InternVL Supported)

Taking InternVL as an example, the data configuration file is `examples/internvl3.5/data.json`. A single dataset is used by default, and the `basic_parameters` format is as follows:

```json
    "basic_parameters": {
        "data_path": "/path/dataset_json_path",
        "data_folder": "/path/dataset_root_path",
        "repeat_time": 1
    }
```

Assuming you want to train two datasets (dataset1 and dataset2), modify `basic_parameters` to the list format as follows:

```json
    "basic_parameters": [{
        "data_path": "/path/dataset1_json_path",
        "data_folder": "/path/dataset1_root_path",
        "repeat_time": 1
    },
    {
        "data_path": "/path/dataset2_json_path",
        "data_folder": "/path/dataset2_root_path",
        "repeat_time": 1
    }]
```

`repeat_time` controls the repeat ratio of the corresponding dataset: when greater than 1, samples are repeated by the corresponding multiple; when less than 1, only the front portion of samples at the corresponding ratio is used.

> [!NOTE]
>
> This multi-dataset method applies to configurations with `dataset_type: multimodal` (such as InternVL). For models using `dataset_type: huggingface` (such as the Qwen series), configure multiple datasets in a comma-separated manner. For details, see [Data Construction for VL Models](./building_data_for_VLModel.md).

## Data Module Addition Process

1.mindspeed_mm/data/data_utils/multimodal_image_video_preprocess.py

Add the image and video preprocessing logic for the corresponding model.

2.mindspeed_mm/data/datasets/multimodal_dataset.py

When `get_item` is called, the returned dictionary is initialized via `_init_return_dict`. Before returning, excess keys are filtered out using `_filter_return_dict_keys`. If you need to return other keys, you must add them additionally in the `_init_return_dict` method.

```shell
def _init_return_dict():
    return {
        "pixel_values": None,
        "image_flags": None,
        "input_ids": None,
        "labels": None,
        "attention_mask": None,
        ...
    }
```

3.mindspeed_mm/data/data_utils/utils.py

Add the preprocessing method for the corresponding model.
