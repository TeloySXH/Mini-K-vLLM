"""
Configuration management for KvChat LLM inference engine.

This module defines the Config dataclass which manages all inference hyperparameters
with comprehensive validation and automatic computation of derived parameters.
"""

import os
import logging
from dataclasses import dataclass
from transformers import AutoConfig

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Inference configuration with comprehensive parameter validation.
    
    Attributes:
        model: Path to model directory (must exist)
        max_num_batched_tokens: Maximum tokens per batch
        max_num_seqs: Maximum concurrent sequences in batch
        max_model_len: Maximum model context length
        gpu_memory_utilization: GPU memory utilization ratio (0, 1]
        tensor_parallel_size: Tensor parallelism degree (1-8)
        enforce_eager: Whether to disable CUDA graph compilation
        hf_config: HuggingFace model config (auto-loaded)
        eos: End-of-sequence token ID
        kvcache_block_size: KV-cache block size (must be 256 multiple)
        num_kvcache_blocks: Number of KV-cache blocks (-1 = auto)
        kv_compress_enabled: Enable KV-cache compression in decode phase
        kv_compress_N: Compression window parameter
        kv_compress_S: Compression trigger threshold (auto-computed)
        kv_compress_R: Compression retention tokens (auto-computed)
        query_window_size: Query window for SnapKV compression
    """
    model: str
    max_num_batched_tokens: int = 50000
    max_num_seqs: int = 1
    max_model_len: int = 50000
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: 'AutoConfig | None' = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # KV compression parameters
    kv_compress_enabled: bool = True
    kv_compress_N: int = 1
    kv_compress_S: int = 0  # Auto-computed
    kv_compress_R: int = 0  # Auto-computed
    query_window_size: int = 50

    def __post_init__(self):
        """Validate all parameters and compute derived values."""
        # 1. Validate model path
        if not os.path.isdir(self.model):
            raise ValueError(
                f"Model path does not exist or is not a directory: {self.model}"
            )
        logger.info(f"Model path validated: {self.model}")
        
        # 2. Validate kvcache_block_size
        if self.kvcache_block_size % 256 != 0:
            raise ValueError(
                f"kvcache_block_size must be a multiple of 256, "
                f"got {self.kvcache_block_size}"
            )
        if self.kvcache_block_size <= 0:
            raise ValueError(f"kvcache_block_size must be positive, got {self.kvcache_block_size}")
        
        # 3. Validate tensor_parallel_size
        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(
                f"tensor_parallel_size must be in range [1, 8], "
                f"got {self.tensor_parallel_size}"
            )
        logger.info(f"Tensor parallelism: {self.tensor_parallel_size}")
        
        # 4. Validate gpu_memory_utilization
        if not (0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"gpu_memory_utilization must be in range (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )
        
        # 5. Validate and load HuggingFace config
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model)
            logger.info(f"Loaded HuggingFace config from {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace config from {self.model}: {e}")
        
        # 6. Validate max_num_seqs
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")
        
        # 7. Validate max_num_batched_tokens
        if self.max_num_batched_tokens <= 0:
            raise ValueError(
                f"max_num_batched_tokens must be positive, "
                f"got {self.max_num_batched_tokens}"
            )
        
        # 8. Adjust max_model_len to model's maximum
        original_max_model_len = self.max_model_len
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings
        )
        if self.max_model_len < original_max_model_len:
            logger.info(
                f"Adjusted max_model_len from {original_max_model_len} "
                f"to {self.max_model_len} (model max: {self.hf_config.max_position_embeddings})"
            )
        
        if self.max_model_len <= 0:
            raise ValueError(
                f"max_model_len must be positive after adjustment, got {self.max_model_len}"
            )
        
        # 9. Validate max_num_batched_tokens >= max_model_len
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be "
                f">= max_model_len ({self.max_model_len})"
            )
        
        # 10. Validate query_window_size
        if self.query_window_size <= 0:
            raise ValueError(
                f"query_window_size must be positive, got {self.query_window_size}"
            )
        
        # 11. Validate kv_compress_N
        if self.kv_compress_N <= 0:
            raise ValueError(f"kv_compress_N must be positive, got {self.kv_compress_N}")
        
        # 12. Auto-compute compression thresholds
        self.kv_compress_S = self.kvcache_block_size * (self.kv_compress_N + 1) - 1
        self.kv_compress_R = self.kvcache_block_size * self.kv_compress_N + 1
        logger.info(
            f"Compression thresholds: S={self.kv_compress_S}, R={self.kv_compress_R}"
        )
        
        # 13. Alias for distributed training compatibility
        # world_size = tensor_parallel_size for compatibility with distributed APIs
        self.world_size = self.tensor_parallel_size
