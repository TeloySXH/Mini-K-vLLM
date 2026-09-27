"""
Neural Network Layers and Components

This module contains optimized layer implementations for LLM inference including:
- Attention mechanisms (with KV-cache support)
- Embeddings and positional encodings
- Normalization layers
- Activation functions
- KV-cache compression algorithms

Components
----------
- attention: Multi-head and grouped query attention
- embed_head: Token and position embeddings
- linear: Optimized linear layers
- layernorm: RMSNorm and LayerNorm implementations
- rotary_embedding: Rotary position embeddings (RoPE)
- sampler: Token sampling for generation
- CompressMethod: KV-cache compression (SnapKV)
- compress_utils: Compression utilities
"""

__all__ = [
    "attention",
    "embed_head",
    "linear",
    "layernorm",
    "rotary_embedding",
    "sampler",
    "CompressMethod",
    "compress_utils",
]
