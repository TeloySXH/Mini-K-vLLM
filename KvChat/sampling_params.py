"""
Sampling parameters for token generation during inference.

This module defines the SamplingParams dataclass which controls how tokens are
sampled during the generation phase of inference (decoding).
"""

from dataclasses import dataclass
from typing import Optional
import logging

from KvChat.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_IGNORE_EOS,
    MIN_TEMPERATURE,
)

logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    """
    Parameters controlling token sampling during generation.
    
    These parameters are applied to each sequence during the decode phase
    to control the randomness and diversity of generated tokens.
    
    Attributes:
        temperature: Sampling temperature controlling randomness.
            - 1.0: Pure probability distribution (no temperature scaling)
            - < 1.0: More deterministic (peaks are sharper)
            - > 1.0: More random (distribution flattens)
            Value must be > 1e-10. Greedy decoding (temperature=0) is not permitted.
        
        max_tokens: Maximum number of tokens to generate per sequence.
            If sequence reaches this limit, generation stops.
        
        ignore_eos: Whether to ignore EOS token and continue generation.
            - True: Continue generating even after EOS
            - False: Stop generation when EOS is encountered
        
        compress_enabled: Optional override for KV-cache compression.
            - None: Use default config setting
            - True: Force compression enabled for this sequence
            - False: Force compression disabled for this sequence
        
        compress_S: Optional override for compression threshold S.
            When sequence KV-context length reaches S, trigger compression.
            - None: Use config value
            - int: Use this custom threshold
        
        compress_R: Optional override for retained tokens after compression.
            How many tokens to retain in KV-cache after compression.
            - None: Use config value
            - int: Use this custom value
    
    Examples:
        >>> # Default sampling
        >>> params1 = SamplingParams()
        
        >>> # Hot temperature for more diverse generation
        >>> params2 = SamplingParams(temperature=1.5, max_tokens=256)
        
        >>> # Cold temperature for more deterministic output
        >>> params3 = SamplingParams(temperature=0.5, max_tokens=512)
        
        >>> # Custom compression settings
        >>> params4 = SamplingParams(compress_enabled=False)
    """
    
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    ignore_eos: bool = DEFAULT_IGNORE_EOS
    compress_enabled: Optional[bool] = None
    compress_S: Optional[int] = None
    compress_R: Optional[int] = None
    
    def __post_init__(self):
        """
        Validate sampling parameters after initialization.
        
        Raises:
            ValueError: If any parameter violates constraints
        """
        # Validate temperature
        if self.temperature <= MIN_TEMPERATURE:
            raise ValueError(
                f"temperature must be > {MIN_TEMPERATURE} "
                f"(greedy sampling is not permitted), got {self.temperature}"
            )
        
        # Validate max_tokens
        if self.max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )
        
        # Validate compression parameters
        if self.compress_S is not None and self.compress_S <= 0:
            raise ValueError(
                f"compress_S must be positive if specified, got {self.compress_S}"
            )
        
        if self.compress_R is not None and self.compress_R <= 0:
            raise ValueError(
                f"compress_R must be positive if specified, got {self.compress_R}"
            )
        
        logger.debug(
            f"SamplingParams initialized: temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, ignore_eos={self.ignore_eos}"
        )

