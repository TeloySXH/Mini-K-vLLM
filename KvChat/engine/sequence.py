"""
Sequence management for LLM inference with thread-safe ID generation.

This module defines sequences which represent individual requests in batched inference,
with support for prefill, decode, and completion tracking.
"""

from copy import copy
from enum import Enum, auto
from typing import List, Optional
import threading
import logging

from KvChat.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class SequenceStatus(Enum):
    """Status of a sequence during inference."""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class _SequenceIDGenerator:
    """Thread-safe sequence ID generator using atomic counter."""
    
    def __init__(self):
        """Initialize with thread lock and counter."""
        self._lock = threading.Lock()
        self._counter = 0
    
    def next_id(self) -> int:
        """
        Generate next sequence ID (thread-safe).
        
        Returns:
            Next unique sequence ID
        """
        with self._lock:
            self._counter += 1
            return self._counter


# Global thread-safe ID generator
_seq_id_generator = _SequenceIDGenerator()


def _get_next_seq_id() -> int:
    """
    Get next sequence ID (internal use).
    
    Returns:
        Next unique sequence ID
    """
    return _seq_id_generator.next_id()


class Sequence:
    """
    Represents a single sequence (request) in LLM inference.
    
    A sequence tracks:
    - Token IDs (prompt + generated completions)
    - Generation progress (prefill vs decode phase)
    - KV-cache state (block allocation, cached tokens)
    - Sampling parameters (temperature, max_tokens, etc.)
    - RoPE position for rotary embeddings
    
    Attributes:
        seq_id: Unique sequence identifier
        status: Current sequence status (WAITING/RUNNING/FINISHED)
        token_ids: List of all token IDs (prompt + completion)
        last_token: Last generated token ID
        num_tokens: Total number of tokens
        num_prompt_tokens: Number of prompt tokens
        num_cached_tokens: Number of tokens with cached KV values
        block_table: Paged attention block allocations
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        ignore_eos: Whether to ignore EOS token
        generated_completion_tokens: Count of generated tokens
        rope_pos: Current RoPE position for rotary embeddings
        need_recompute: Flag for recomputation requirement
    """
    
    BLOCK_SIZE = 256  # Standard KV-cache block size

    def __init__(self, token_ids: List[int], sampling_params: Optional[SamplingParams] = None):
        """
        Initialize a new sequence.
        
        Args:
            token_ids: Initial token IDs (prompt)
            sampling_params: Sampling parameters (uses defaults if None)
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        self.seq_id = _get_next_seq_id()
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.generated_completion_tokens = 0
        self.rope_pos = self.num_tokens - 1
        self.need_recompute = False
        self.block_size = self.BLOCK_SIZE
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        
        self.generated_completion_tokens += 1
        self.rope_pos += 1
        self.need_recompute = False
    def __getstate__(self):

        return {
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "block_table": self.block_table,
            "token_ids": self.token_ids,
            "last_token": getattr(self, "last_token", None),
            "generated_completion_tokens": getattr(self, "generated_completion_tokens", 0),
            "rope_pos": getattr(self, "rope_pos", self.num_tokens - 1),
            "seq_id": getattr(self, "seq_id", None),#!!!新增
        }

    def __setstate__(self, state):

        self.num_tokens = state.get("num_tokens")
        self.num_prompt_tokens = state.get("num_prompt_tokens")
        self.num_cached_tokens = state.get("num_cached_tokens")
        self.block_table = state.get("block_table", [])

        token_ids = state.get("token_ids", None)
        last_token = state.get("last_token", None)

        if token_ids is not None:
            self.token_ids = token_ids
            if not isinstance(self.token_ids, list):
                self.token_ids = [self.token_ids]
            if self.token_ids:
                self.last_token = self.token_ids[-1]
            else:
                self.last_token = last_token
        else:
            if last_token is None:
                self.last_token = 0
            else:
                self.last_token = last_token
            self.token_ids = [self.last_token]
        self.generated_completion_tokens = state.get("generated_completion_tokens", 0)
        self.rope_pos = state.get("rope_pos", self.num_tokens - 1)

        self.seq_id = state.get("seq_id", None)#!!!新增