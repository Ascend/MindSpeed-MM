# Seqpack

## Background and Challenges

When training multimodal large models, the input sequence length exhibits high heterogeneity due to variations in the number of image tokens and text length. Traditional methods require padding to the maximum length within a batch, leading to memory waste. Furthermore, traditional methods do not account for the relationship between the number of tokens across DP groups, which can easily introduce load imbalance across cards.

## Solution

Multiple sequences are concatenated to a length approximating `max-seq-len`, and the concatenated data is fed into the model as a single batch. The model processes the concatenated data in a TND layout mode. To ensure a sufficient number of samples are available for sequence concatenation, a buffer is used to store data. During the concatenation process, sequences are popped from the buffer to form batches with a total length approximating `max_seq_len`. This approach ensures a consistent total token count per card, saving the GPU memory otherwise wasted on padding while balancing the data load across cards.

## How to Use

Currently, seqpack supports the Qwen3 VL model. The following parameters can be set in the `gpt_args` section of the corresponding model configuration file:

```shell
gpt_args:
    ....
    use_txt_dynamic_batching: true
    max_seq_len: MAX_SEQ_LEN
    dynamic_batch_buffer_size: BUFFER_SIZE
```

Where:

* `use_txt_dynamic_batching`: Switch for seqpack. Setting it to `true` enables the seqpack feature. The default value is `false`.
* `max_seq_len`: Sets the length of the concatenated sequence. The default value is `2048`.
* `dynamic_batch_buffer_size`: Buffer size. The default value is `200`.
