# Dynamic DP/CP Switch

## Problem Analysis

During training on dynamic datasets, input data sizes vary across iterations, with longer sequences prone to out-of-memory (OOM) errors. The current MindSpeed MM framework employs a static parallelism strategy, determining DP/CP degrees based on the maximum sequence length in the dataset during initialization to ensure trainability for long sequences. However, this static approach often leads to suboptimal performance. In scenarios where long and short sequences are mixed, CP is often set relatively high to accommodate long sequences, forcing short sequences to also use CP, which reduces their computational efficiency. When long sequences are few, short sequences are many, and the gap between long and short sequences is large, the performance degradation can be significant.

## Solution

For each training iteration, determine the parallelism strategy (i.e., DPxCPy) based on the data obtained by each DP process, then perform data distribution and parallel group switching accordingly. This ensures both trainability for long sequences and computational efficiency for short sequences.

- Parallel group initialization: After megatron initialization, a list of parallel groups is automatically generated using the user-configured maximum CP size. For example, if `CP=4`, the parallel groups can be `{dp4cp1, dp2cp2, dp1cp4}`.

- New parallel strategy acquisition: When dynamic DP/CP is enabled, the data loader fetches data in DPnCP1 mode by default. For each iteration, gather data sizes across all DP domains and traverse the parallel groups to select the optimal strategy (prioritizing DP) that meets the sequence length requirements.

- Parallel domain switching and data distribution: After obtaining the new parallel strategy, update the global variables of the parallel group to point to the new group. During the switch from DP to CP, data broadcasting is required. Cards that receive the broadcast need to cache the samples they have already obtained, and in subsequent training steps, prioritize fetching sample data from this cache.

## Compatible Version

OpenSoraPlan-1.3

## How to Use

Parameter location: `pretrain_t2v_model.json`

| Parameter Name              | Description                                            |
| --------------------------- | -------------------------------------------------- |
| `use_dynamic_dpcp`         | Switch for the dynamic DPCP feature. Currently only supports Ulysses sequence parallelism and does not yet support encoder DP/interleaved offload.                              |
| `max_cp_size`               | Maximum CP size supported by the cluster. For example, if `max_cp_size=4`, the parallel groups can be `{dp4cp1, dp2cp2, dp1cp4}`.|
| `max_seq_size`              | Maximum sequence length that a single card can compute. For example, the longest video might be (23 x 1080 x 720) (fps, height, width).|
