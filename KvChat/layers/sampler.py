import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        [OPTIMIZATION] Numerically stable sampling with complete robustness
        
        Key improvements:
        1. Clamp logits to prevent overflow
        2. Clamp temperatures to reasonable range
        3. Clamp probabilities to valid range
        4. Fallback to greedy sampling if multinomial fails
        """
        # 1. Type conversion
        logits = logits.float()
        temperatures = temperatures.float()
        
        # 2. Numerical stability: subtract max for numerical stability
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        logits = logits - logits_max
        
        # 3. Clamp logits to prevent exponential overflow
        logits = torch.clamp(logits, min=-1e8, max=1e8)
        
        # 4. Clamp temperatures to reasonable range (prevent extreme values)
        temperatures = torch.clamp(temperatures, min=1e-2, max=100.0)
        
        # 5. Temperature scaling
        logits = logits / temperatures.unsqueeze(dim=1)
        
        # 6. Softmax
        probs = torch.softmax(logits, dim=-1)
        
        # 7. Clamp probabilities to valid range
        probs = torch.clamp(probs, min=1e-10, max=1.0 - 1e-10)
        
        # 8. Verify and correct probability sums if needed
        prob_sum = probs.sum(dim=-1, keepdim=True)
        if (prob_sum < 0.99).any() or (prob_sum > 1.01).any():
            probs = probs / prob_sum  # Renormalize
        
        # 9. Multinomial sampling with fallback
        try:
            sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        except RuntimeError as e:
            # Fallback to greedy sampling if multinomial fails
            logger.warning(f"Multinomial sampling failed: {e}, falling back to argmax")
            sample_tokens = torch.argmax(probs, dim=-1)
        
        return sample_tokens
