import torch
from torch import nn
import triton
import triton.language as tl
import os
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from KvChat.utils.context import get_context
from KvChat.layers.compress_utils import MyCompressCompact,store_qkvcache

# ==================== Store KV Cache Kernels ====================

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    
    [OPTIMIZATION] Enhanced kernel design for better memory efficiency.
    Processes KV pairs by token and head to optimize cache utilization.
    """
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    if slot_idx == -1:
        return
    
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    head_idx = tl.program_id(1)
    
    head_offsets = tl.arange(0, head_dim)
    input_offset = (token_idx * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)

    cache_offset = (block_idx * block_size * num_kv_heads * head_dim +
                   block_offset * num_kv_heads * head_dim +
                   head_idx * head_dim +
                   head_offsets)
    
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 16
):
    """
    Store key-value pairs into paged cache.
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim) - Query keys to store
        value: (num_tokens, num_kv_heads, head_dim) - Query values to store
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim) - Paged K cache
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim) - Paged V cache
        slot_mapping: (num_tokens,) - Maps each token to a cache slot
        block_size: Number of tokens per block (default: 16)
    
    Returns:
        None (modifies k_cache and v_cache in-place)
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # Ensure tensors are contiguous for efficient kernel access
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    # Validate shapes and consistency
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"
    assert k_cache.shape[2] == num_kv_heads, "Cache num_kv_heads mismatch"
    assert k_cache.shape[3] == head_dim, "Cache head_dim mismatch"
    
    grid = (num_tokens, num_kv_heads)
    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


class Attention(nn.Module):
    """
    Flash Attention module with paged KV cache support and compression.
    
    Supports two attention phases:
    - Prefill: Processes full sequences with variable lengths
    - Decode: Processes one token at a time with paged KV cache
    
    Features:
    - Flash Attention for efficient computation
    - Paged KV cache for memory efficiency
    - KV cache compression for long sequences
    - Optional GQA (Grouped Query Attention) support
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        vllm_config,
        num_layers,
        block_size: int = 16
    ):
        """
        Initialize Attention module.
        
        Args:
            num_heads: Number of query attention heads
            head_dim: Dimension of each attention head
            scale: Softmax scale factor (typically 1/sqrt(head_dim))
            num_kv_heads: Number of key-value heads (for GQA)
            vllm_config: Configuration object with compression settings
            num_layers: Total number of transformer layers (for compression)
            block_size: Size of each KV cache block in tokens (default: 16)
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.block_size = block_size  # Block size for paged cache

        # KV Cache Compression configuration
        self.kv_compress_enabled = vllm_config.kv_compress_enabled
        self.kv_compress_N = vllm_config.kv_compress_N
        self.kv_compress_S = vllm_config.kv_compress_S
        self.kv_compress_R = vllm_config.kv_compress_R
        if self.kv_compress_enabled:
            self.query_window_manager = None
            self.query_window_size = vllm_config.query_window_size
            self.num_layers = num_layers

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, Layer):
        """
        Forward pass using Flash Attention with paged KV cache.
        
        Args:
            q: Query tensor
                - Prefill: (total_tokens, num_heads, head_dim)
                - Decode: (batch_size, num_heads, head_dim)
            k: Key tensor
                - Prefill: (total_tokens, num_kv_heads, head_dim)
                - Decode: (batch_size, num_kv_heads, head_dim)
            v: Value tensor
                - Prefill: (total_tokens, num_kv_heads, head_dim)
                - Decode: (batch_size, num_kv_heads, head_dim)
            Layer: Current transformer layer index
        
        Returns:
            Attention output with same shape as input query (q)
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store k, v into paged cache during prefill
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.block_size)

        # KV Cache Compression in decode phase (if enabled)
        if not context.is_prefill and self.kv_compress_enabled:
            if self.query_window_manager is not None:
                active_indices = context.q_window_active_indices
                active_seq_ids = context.q_window_active_seq_ids
                if active_indices:
                    idx_tensor = torch.tensor(active_indices, device=q.device, dtype=torch.long)
                    q_active = q.index_select(0, idx_tensor)  # [m, q_heads, head_dim]
                    self.query_window_manager.append(active_seq_ids, Layer, q_active)
            if context.compress_any:
                runtime_S = context.runtime_compress_S if context.runtime_compress_S > 0 else self.kv_compress_S
                runtime_R = context.runtime_compress_R if context.runtime_compress_R > 0 else self.kv_compress_R

                MyCompressCompact(
                    query_window_manager=self.query_window_manager,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    layer_id=Layer,
                    block_size=k_cache.size(1),
                    S=runtime_S,
                    R=runtime_R,
                    query_window_size = self.query_window_size,
                    context = context,
                    num_layers = self.num_layers
                )   

        # Compute attention using flash_attn library
        if context.is_prefill:
            # Prefill phase: use variable-length flash attention
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            
            # [OPTIMIZATION] Flash Attention with variable sequence lengths
            # Supports efficient batched prefill with different sequence lengths
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
            
        else:
            # Decode phase: use cached KV with single query token
            # [OPTIMIZATION] Flash Attention with paged KV cache
            # One query token at a time, attending to full cached sequence
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),  # Add sequence dimension for single token
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        
        return o

