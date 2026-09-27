"""
KvChat: Efficient KV-Cache Compression for Fast LLM Inference

An optimized LLM inference engine with advanced features:
- Memory-efficient KV-cache management via SnapKV compression
- Multi-batch token processing with paged attention
- Support for tensor parallelism (1-8x)
- Thread-safe inference context management
- Comprehensive error handling and logging

Quick Start
-----------

Basic usage:

    >>> from KvChat import LLM, SamplingParams
    >>> from KvChat.config import Config
    >>> 
    >>> # Create configuration
    >>> config = Config(
    ...     model="./model_dir",
    ...     max_num_batched_tokens=50000,
    ...     max_model_len=50000,
    ... )
    >>> 
    >>> # Initialize inference engine
    >>> llm = LLM(config)
    >>> 
    >>> # Generate tokens with custom sampling
    >>> prompt_tokens = [1, 2, 3, 4, 5]
    >>> params = SamplingParams(temperature=0.7, max_tokens=256)
    >>> outputs = llm.generate(prompt_tokens, params)

Key Components
--------------

- **LLM**: Main inference interface
- **Config**: Inference configuration with parameter validation
- **SamplingParams**: Token generation parameters
- **logging_config**: Centralized logging management
- **constants**: Global configuration constants

Features
--------

KV-Cache Compression
    SnapKV algorithm reduces memory overhead by:
    - Detecting important tokens via attention scores
    - Retaining only essential KV values
    - Maintaining generation quality

Paged Attention
    Efficient memory management through:
    - Block-based KV-cache allocation
    - Automatic block table management
    - Reduced fragmentation

Multi-GPU Support
    - Tensor parallelism (1-8 GPUs)
    - Distributed batch processing
    - Coordinated inference

Configuration
    >>> from KvChat.config import Config
    >>> config = Config(
    ...     model="path/to/model",
    ...     tensor_parallel_size=2,
    ...     gpu_memory_utilization=0.8,
    ...     kv_compress_enabled=True,
    ...     max_num_seqs=32,
    ... )

Logging
    >>> from KvChat.logging_config import setup_logging
    >>> setup_logging(log_file="inference.log")

References
----------

Paper: SnapKV - Efficient KV-Cache Compression with Attention-Based Token Selection

Version: 1.0.0
"""

from KvChat.llm import LLM
from KvChat.sampling_params import SamplingParams
from KvChat.config import Config

__version__ = "1.0.0"
__author__ = "Mini-KS Team"

__all__ = [
    "LLM",
    "Config", 
    "SamplingParams",
]

