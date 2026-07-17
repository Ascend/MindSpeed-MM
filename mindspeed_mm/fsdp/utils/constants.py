# some constants used for dataloader
# PrefetchGradAccDataLoader
GLOBAL_STEP_TOKEN_NUM = "global_step_token_num"  # tokens number per step per rank
AVG_PER_STEP_TOKEN_NUM = "avg_per_step_token_num"  # average tokens number per step per rank (gcs in consider)

# Memory report step, after optimizer state has been initialized.
MEMORY_REPORT_ITERATION = 2

# Constants used for checkpoint weight conversion.
LATEST_TXT = "latest_checkpointed_iteration.txt"
DCP_CHECKPOINT_VERSION = 3.0
DIR_MODE = 0o750
FILE_MODE = 0o640
