# Seqpack

## 背景与挑战

在训练多模态大模型时，输入序列长度因图像token数量和文本长度的差异呈现高度异构性。传统的方式需按批次内最大长度进行Padding，导致显存浪费。此外，传统方式并未考虑DP组间token数量的关系，容易引入卡间负载不均的问题。

## 解决方案

将多条序列拼接成近似于`max-seq-len`的长度，并将拼接后的数据作为一个批次数据输入模型，模型以TND的layout模式处理拼接后的数据。为确保有足够可选择的样本用于序列拼接，采用`buffer`存储数据，在拼接序列过程中，从`buffer`内弹出能够拼接成长度近似于`max-seq-len`的序列批次。这样一来，每张卡上的token总数一致，在节约padding的显存的同时，均衡卡间数据负载。

## 使用方法

当前Seqpack支持Qwen3 VL模型，可在对应的模型配置文件中的`gpt_args`部分设置如下参数：

```shell
gpt_args:
    ....
    use_txt_dynamic_batching: true
    max_seq_len: MAX_SEQ_LEN
    dynamic_batch_buffer_size: BUFFER_SIZE
```

其中，

* `use_txt_dynamic_batching`: seqpack的开关，设置为`true`视为开启seqpack功能，默认值为`false`;
* `max_seq_len`: 设定拼接后序列的长度，默认值为`2048`;
* `dynamic_batch_buffer_size`: `buffer`的大小，默认值为`200`。
