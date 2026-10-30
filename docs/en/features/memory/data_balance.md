# Data balance

## 1. Background

Due to factors such as native resolution, multimodal models suffer from unbalanced data load in distributed training, which reduces cluster resource utilization.

The Data Balance module is designed to solve the problem of unbalanced data load. Through online data rearrangement, it achieves load balancing across DP and improves resource utilization.

## 2. Solution

Note: ViT data load balancing is currently supported. The LLM part does not use the packing strategy. LLM packing will be supported later, which, combined with the current online load balancing scheme for the ViT part, to improve the resource utilization of the entire network.

### 2.1 Key Components

| Component | Description |
|------|------|
| **DataBalance class** | Main class that coordinates the data balancing process |
| **Sorting algorithm** | Such as `post_global_balancing_greedy_without_pad` (more sorting algorithms will be supported later) |
| **PrefetchMicroBatchIterator** | Prefetches data asynchronously to hide latency |
| **Helper functions** | Functions for data splitting, mapping, reorganization, and communication |

### 2.2 Execution Flow

```mermaid
flowchart TD
    A[Start training] --> B[Get the original batch (GBS)]
    B --> C[Obtain data lengths across DP]
    C --> D[Reorder data indices for load balancing]
    D --> E[Ranktable mapping]
    E --> F[All2All data distribution]
    F --> G[Build the load-balanced iterator]
    G --> H[Training step]
    H --> I[Create the load-balanced iterator]
    I --> J[Forward/backward pass and gradient update]
    J --> K{Next iteration?}
    K -->|Yes| B
    K -->|No| L[End training]
```

### 2.2.1 Data Distribution Example

"Purple" indicates the initial state and "blue" indicates the target state. Data is redistributed through data mapping paths and All2All communication.

![all2all](../../../../sources/images/features/online-data-balance/all2all.png)

## 3. Usage Guide

### 3.1 Enabling the Configuration

Add the following to the Qwen 2.5 Omni model training command:

```bash
--use-data-balance
```

### 3.2 Core APIs

#### 3.2.1 DataBalance Initialization

```python
from mindspeed_mm.utils.data_balance.data_balance import DataBalance

data_balancer = DataBalance(
    virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
    model_config_path=args.model_config_path,
    sorting_algo_name=args.sorting_algo_name,
    len_model=len_model,
    train_data_iterator=train_data_iterator
)
```

#### 3.2.2 Creating a Load-Balanced Data Iterator

```python
batch_generator = data_balancer.build_balanced_train_data_iterator(
    is_vit_last_stage=is_vit_last_stage,
    max_batch_capacity=max_batch_capacity,
    micro_batch_size=micro_batch_size,
    num_microbatches=num_microbatches,
    data_type='image'
)

# Use it in the training loop
for batch in batch_generator:
    # Training step
    ...
```

## 4. Best Practices

- **Algorithm selection**: `post_global_balancing_greedy_without_pad` is used by default (more algorithms will be supported later).
- **Parallel configuration**: Ensure that the DP degree of the image encoder is greater than 1.

## 5. Performance Gain

- **Training speed**: computation load balancing shortens the training time, delivering a gain of 5%+ in typical scenarios.
