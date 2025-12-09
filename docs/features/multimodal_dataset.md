# MultiModal Dataset

## 多数据集训练
### 使用方法（InternVL已支持）
以internvl为例，在examples/internvl2.5/data_4B.json中，修改basic_parameters。

假设要训练dataset1和dataset2两个数据集，修改如下
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

## 理解模型数据模块添加流程
1.mindspeed_mm/data/data_utils/multimodal_image_video_preprocess.py

添加对应模型的图像和视频预处理逻辑


2.mindspeed_mm/data/datasets/multimodal_dataset.py

在get_item时，会通过_init_return_dict初始化返回的字典，return前通过_filter_return_dict_keys过滤多余的key。如果需要返回其余的key，需要在_init_return_dict方法中额外添加
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

添加对应模型的preprocess方法
