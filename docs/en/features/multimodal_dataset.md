# Multimodal Dataset

## Multi-Dataset Training

### How to Use (InternVL Supported)

Taking InternVL as an example, modify `basic_parameters` in `examples/internvl2.5/data_4B.json`.

Assuming you want to train two datasets (dataset 1 and dataset 2), modify as follows:

```shell
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

## Data Module Addition Process

1. `mindspeed_mm/data/data_utils/multimodal_image_video_preprocess.py`

   Add the image and video preprocessing logic for the corresponding model.

2. `mindspeed_mm/data/datasets/multimodal_dataset.py`

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

3. `mindspeed_mm/data/data_utils/utils.py`

Add the preprocessing method for the corresponding model.
