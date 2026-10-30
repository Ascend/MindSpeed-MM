# Dynamic DP/CP switch

## Problem Analysis

During training on dynamic datasets, the input data size varies from iteration to iteration, and longer sequences are prone to OOM. The current MM framework uses a static parallel strategy that determines the DP/CP parallel degrees at initialization based on the longest size in the dataset, so that long sequences can be trained. However, this static approach tends to yield suboptimal performance. In scenarios with mixed long and short sequences, CP is often set relatively large to accommodate long sequences, so short sequences are also processed with CP parallelism, which reduces the computational efficiency of short sequences. In scenarios where long sequences are few, short sequences are many, and the gap between them is large, the performance degradation is severe.

## Solution

For each training iteration, determine the parallelism strategy (i.e., DPxCPy) based on the data obtained by each DP process, then perform data distribution and parallel group switching accordingly. This ensures both trainability for long sequences and computational efficiency for short sequences.

- Parallel group initialization: After megatron initialization, a list of parallel groups is automatically generated using the user-configured maximum CP size. For example, if `CP=4`, the parallel groups can be `{dp4cp1, dp2cp2, dp1cp4}`.

- New parallel strategy acquisition: When dynamic DP/CP is enabled, the data loader fetches data in DPnCP1 mode by default. For each iteration, gather data sizes across all DP domains and traverse the parallel groups to select the optimal strategy (prioritizing DP) 
that meets the sequence length requirements.

- Parallel domain switching and data distribution: After obtaining the new parallel strategy, update the global variables of the parallel group to point to the new group. During the switch from DP to CP, data broadcasting is required. Cards that receive the broadcast need to cache the samples they have already obtained, and in subsequent training steps, prioritize fetching sample data from this cache.

## Compatible Version

Open-Sora Plan v1.3

## How to Use

Parameter location: `pretrain_t2v_model.json`

| Parameter Name              | Description                                            |
| --------------------------- | -------------------------------------------------- |
| `use_dynamic_dpcp`         | Switch for the dynamic DPCP feature. Currently only supports Ulysses sequence parallelism and does not yet support encoder DP/interleaved offload.                              |
| `max_cp_size`               | Maximum CP size supported by the cluster. For example, if `max_cp_size=4`, the parallel groups can be `{dp4cp1, dp2cp2, dp1cp4}`.|
| `max_seq_size`              | Maximum sequence length that a single card can compute. For example, the longest video might be (23 x 1080 x 720) (FPS, height, width).|
