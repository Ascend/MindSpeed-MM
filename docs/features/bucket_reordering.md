## 数据负载均衡(数据分桶重排序)

### 数据分桶训练

对数据进行分桶重排序，使得数据层的负载达到更好的均衡。

数据负载均衡的方案分为两种：

​	1. 数据分桶：性能优先，"priority_mode"配置为 "data_bucketing_img"，若不配置，默认为数据分桶；

​	2. 数据重排：精度优先，"priority_mode"配置为 "data_reordering_img"



### 使用方法（Qwen2VL 已支持）

#### Qwen2VL的数据分桶使用方法：

在examples/qwen2vl/data_2b.json中，修改dataloader_param下的sampler_type为"BucketBatchSampler"，且"priority_mode"配置为 "data_reordering_img"，如下：

	    "dataloader_param": {
	        "dataloader_mode": "sampler",
	        "drop_last": true,
	        "sampler_type": "BucketBatchSampler",
	        "priority_mode": "data_reordering_img",
	        "collate_param": {
	            "model_name": "qwen2vl",
	            "ignore_pad_token_for_loss": true
	        },
	        "pin_memory": true,
	        "data_sharding": true,
	        "shuffle": true
	    }