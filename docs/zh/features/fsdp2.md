# FSDP2

## 背景与挑战

PyTorch的完全分片数据并行（FSDP）旨在提供一个高性能的即时执行模式实现，包含通信分桶和通信/计算掩盖功能。该API通过将一组参数展平拼接成FlatParameter来表示通信桶。然而，这种FlatParameter设计导致难以对桶内单个参数实施差异化操作（如参数冻结、精度转换等），损害了组合灵活性，同时也使内部实现复杂化（例如状态字典逻辑长达数千行代码且需要额外通信）。

## 解决方案

基于上述局限性，FSDP2移除了FlatParameter，采用沿0维分片的DTensor表示分片参数，支持对单个参数的便捷操作、免通信的分片状态字典，以及更简化的初始化流程。

## 使用方法

在mindspeed中FSDP2的入口是一个配置文件，通过生成配置文件，传入命令行参数即可使用该特性。
```
export CUDA_DEVICE_MAX_CONNECTIONS=2 # 设置不能为1
--use-torch-fsdp2 \
--fsdp2-config-path ./fsdp2_config.yaml \
--ckpt-format torch_dcp \
--untie-embeddings-and-output-weights \
# 注意不能打开分布式优化器
```
### 参数详解

fsdp2_config.yaml的配置项如下：

- **`sharding_size`**
  - 描述：控制模型并行度，用于张量分片，默认为1
  - 取值：
    - `"auto"`：自动根据可用设备数量确定最优分片大小
    - 整数值：手动指定分片组的大小

- **`sub_modules_to_wrap`**
  - 描述：指定使用FSDP进行参数分片的子模块
  - 配置格式
    - 使用点号分隔的完整模块路径
    - 从模型的第一层子模块开始（不包含最外层模型变量名）
    - 支持精确路径和模式匹配
  - 示例:
    - `model.model.deepstack_merger_list.{*}`
    - `model.model.language_model.layers.{0-20,22-40}`
    - `model.lm_head`

- **`ignored_modules`**
  - 描述：排除FSDP管理的模块类列表
  - 配置格式：与 **`sub_modules_to_wrap`** 一致

- **`recompute_modules`**
  - 描述：配置激活值重计算，以计算换内存
  - 配置格式：与 **`sub_modules_to_wrap`** 一致
  - 约束：与megatron中的完全重计算功能存在冲突，需将其关闭

- **`use_reentrant`**
  - 描述：选择检查点实现的类型，是否可重入，默认True
  - 取值：`True`或`False`

- **`reshard_after_forward`**
  - 描述：控制何时重新聚合分片参数
  - 取值：
    - `True`：前向传播后立即重新分片参数（节省内存，ZeRO3）
    - `False`：保持参数聚合直到反向传播（更好性能，ZeRO2）

- **`param_dtype`**
  - 描述：参数存储和计算的数据类型
  - 取值：`"bf16"`，`"fp16"`，`"fp32"`

- **`reduce_dtype`**
  - 描述：梯度规约操作的数据类型
  - 取值：`"bf16"`，`"fp16"`，`"fp32"`

- **`output_dtype`**
  - 描述：指定前向输出的数据类型
  - 取值：`"bf16"`，`"fp16"`，`"fp32"`


- **`cast_forward_inputs`**
  - 描述：控制前向传播输入的自动类型转换
  - 取值：`True`或`False`

- **`num_to_forward_prefetch`**
  - 描述：前向传播期间预取参数的后续层数

- **`num_to_backward_prefetch`**
  - 描述：反向传播期间预取参数的后续层数

- **`offload_to_cpu`**
  - 描述：是否将参数，梯度和优化器状态卸载到CPU内存
  - 取值：`True`或`False`，默认为`False`

- **`pin_memory`**
  - 描述：是否锁定CPU内存以提高数据传输效率，只有当开始`offload_to_cpu`才生效
  - 取值：`True`或`False`

### 配置实例
```
sharding_size: auto
sub_modules_to_wrap:
  - "text_decoder.output_layer"
  - "text_decoder.embedding"
  - "text_decoder.rotary_pos_emb"
  - "text_decoder.decoder.layers.{*}"
param_dtype: "bf16"
reduce_dtype: "fp32"
cast_forward_inputs: True
ignored_modules:
  - "image_encoder"
recompute_modules:
  - "text_decoder.decoder.layers.{*}"
num_to_forward_prefetch: 2
num_to_backward_prefetch: 2
offload_to_cpu: False
```

## 使用效果
针对Llama-7B，FSDP2相比FSDP1实现了更高的MFU，峰值内存降低7%，且保持相同的损失曲线。

## 注意事项
1、当开启fsdp2训练时，需关闭分布式优化器及其相关配置

2、当开启fsdp2训练时，模型权重的保存格式`ckpt-format`仅支持`torch_dist`或`torch_dcp`

- 配置为`torch_dist`，模型需通过继承`MegatronModule`或自定义来实现`sharded_state_dict()`方法；同时需保证模型中所有权重的0维size均大于或等于sharding_size

- 配置为`torch_dcp`，模型需通过继承`MegatronModule`或自定义来实现`state_dict_for_save_checkpoint()`方法，并且其返回的权重字典需要与`model.state_dict()`的返回值一致

3、当开启fsdp2训练时，需关闭重计算的相关配置，包括：`--recompute-granularity`、`--recompute-method`以及`--recompute-num-layers`等

4、当设置`offload_to_cpu=True`时，需在入口脚本中设置通信组为双后端，即：`--distributed-backend npu:hccl,cpu:gloo`

5、对于参数量极大的模型训练，推荐启用`--init-model-with-meta-device`和`--no-initialization`参数，有效避免因一次性加载完整模型参数导致的内存溢出，同时显著减少模型初始化阶段的等待时间。

6、fsdp2的混精在yaml中配置生效，`--bf16`不再必要，并且与断点续训存在冲突，如启用需要与`--no-save-optim`和`--no-load-optim`一同启用。
为了和`--bf16`的计算行为对齐，我们增加了`--downcast-to-bf16`选项，在权重加载阶段增加权重downcast，以保持关闭`--bf16`时的计算一致性。
此处fsdp2混精默认保持加载权重的精度不变，推荐使用此默认行为，避免精度损失。