"""Names shared by the vocabulary-parallel modules and the checkpoint code that stores their state."""

# Substring that identifies the host-resident embedding in ``model._no_placement_params``. Qwen3.8's entry is
# ``ple.ple_embedding.ngram_embedding.weight``; the hint is the part such entries share (see
# set_host_resident).
HOST_RESIDENT_HINT = "ngram_embedding"

# Suffix of a host embedding's checkpoint entries. The embedding is an ordinary parameter, not a DTensor, so every
# rank owns a row window of the same shape and a different content; the window index in the key keeps those
# entries apart (see vocab_checkpoint.py).
SHARD_KEY_SUFFIX = ".vp_shard_{:05d}_of_{:05d}"

# Name of the cpu embedding optimizer inside MultiOptimizer. The embeddings keep their own sparse optimizer, so
# its moments are stored under this entry next to the dense ones (see optimizer.py).
CPU_SPARSE_KEY = "cpu_sparse"
