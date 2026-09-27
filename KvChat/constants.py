"""
Global constants for KvChat LLM inference engine.

This module centralizes configuration constants for easier maintenance and IDE support.
"""

# =============================================================================
# KV-Cache Configuration Constants
# =============================================================================

# Standard KV-cache block size (must be multiple of 256)
KVCACHE_BLOCK_SIZE = 256

# Default number of tokens to keep during compression
NUM_KEEP_TOKENS = 220

# Default query window size for SnapKV compression
DEFAULT_QUERY_WINDOW = 50

# Default compression window parameter
DEFAULT_COMPRESS_WINDOW = 5


# =============================================================================
# Model Configuration Constants
# =============================================================================

# Default maximum model context length
DEFAULT_MAX_MODEL_LEN = 50000

# Default maximum tokens per batch
DEFAULT_MAX_BATCHED_TOKENS = 50000

# Minimum tensor parallelism degree
MIN_TENSOR_PARALLEL = 1

# Maximum tensor parallelism degree
MAX_TENSOR_PARALLEL = 8

# Default tensor parallelism
DEFAULT_TENSOR_PARALLEL = 1


# =============================================================================
# CUDA Graph Configuration Constants
# =============================================================================

# Batch sizes for which to compile CUDA graphs
CUDAGRAPH_BATCH_SIZES = (1, 2, 4, 8)

# Default step for CUDA graph batch sizes
CUDAGRAPH_BATCH_STEP = 16


# =============================================================================
# Sampling Configuration Constants
# =============================================================================

# Default sampling temperature (1.0 = no temperature scaling)
DEFAULT_TEMPERATURE = 1.0

# Default maximum tokens to generate per request
DEFAULT_MAX_TOKENS = 512

# Minimum temperature for sampling
MIN_TEMPERATURE = 1e-10

# Default whether to ignore EOS token
DEFAULT_IGNORE_EOS = False


# =============================================================================
# Device Configuration Constants
# =============================================================================

# Default GPU memory utilization ratio
DEFAULT_GPU_MEMORY_UTIL = 0.8

# Maximum GPU memory utilization ratio
MAX_GPU_MEMORY_UTIL = 1.0

# Minimum GPU memory utilization ratio
MIN_GPU_MEMORY_UTIL = 0.01


# =============================================================================
# Inference Configuration Constants
# =============================================================================

# Default number of maximum concurrent sequences
DEFAULT_MAX_SEQ = 1

# Special token IDs
DEFAULT_EOS_TOKEN_ID = -1

# Default block table size
DEFAULT_BLOCK_TABLE_SIZE = 1024


# =============================================================================
# Logging Configuration Constants
# =============================================================================

# Default log file name
DEFAULT_LOG_FILE = "kvchat.log"

# Maximum log file size (10 MB)
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024

# Number of backup log files to keep
NUM_LOG_BACKUPS = 10
