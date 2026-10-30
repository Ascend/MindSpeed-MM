# online-data-balance

## 1. Background

Due to factors such as the native resolution of multimodal models, the data load is imbalanced during distributed training, and cluster resource utilization decreases.

The Data Balance module is designed to solve the problem of imbalanced data load. Through online data rearrangement, it achieves load balancing across DP and improves resource utilization.

## 2. Solution

Note: Currently, ViT data load balancing is supported, and the LLM part does not use the packing policy. LLM packing will be supported later. Combined with the current online load balancing scheme for the ViT part, it will improve the resource utilization of the entire network.

### 2.1 Key Components

**Key Component Description**

| Component | Description |
| --- | --- |
| **DataBalance class** | Main class that coordinates the data balancing process |
| **Sorting algorithm** | For example, `post_global_balancing_greedy_without_pad` (more sorting algorithms will be supported later) |
| **PrefetchMicroBatchIterator** | Asynchronously prefetches data to hide latency |
| **Auxiliary functions** | Data splitting, mapping, reorganization, and communication |

### 2.2 Execution Process

```mermaid
flowchart TD
    A[Start training] --> B[Obtain the original batch (GBS)]
    B --> C[Obtain data lengths across DP]
    C --> D[Rearrange data indexes for load balancing]
    D --> E[Ranktable mapping]
    E --> F[All2All data distribution]
    F --> G[Build the load balancing iterator]
    G --> H[Training step]
    H --> I[Create the load balancing iterator]
    I --> J[Forward and backward passes, gradient update]
    J --> K{Next iteration?}
    K -->|Yes| B
    K -->|No| L[End training]
```

#### 2.2.1 Data Distribution Example

Purple indicates the initial state, and blue indicates the target state. Data rearrangement is implemented through data mapping paths and All2All communication.

<img src="../../../../sources/images/features/online-data-balance/all2all.png" alt="all2all">

## 3. Usage Guide

### 3.1 Enabling the Configuration

Add the following to the training command of the Qwen 2.5 Omni model:

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

#### 3.2.2 Creating a Load Balancing Data Iterator

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

- **Training speed**: Load balancing computation shortens the training time, with a typical performance gain of 5%+.
