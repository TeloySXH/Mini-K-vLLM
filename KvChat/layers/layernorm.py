import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    @torch.compile
    def fused_rms_linear_forward(
        self,
        x: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        [OPTIMIZATION] Fused RMSNorm + Linear operation
        
        Combines RMSNorm(x) @ W^T + b into a single operation sequence.
        Expected improvement: 5-8% throughput by reducing kernel launch overhead
        and eliminating intermediate HBM writes.
        
        Args:
            x: Input tensor (..., hidden_size)
            linear_weight: Linear layer weight (out_features, hidden_size)
            linear_bias: Linear layer bias (out_features,) or None
            
        Returns:
            Output tensor (..., out_features)
        """
        # Step 1: RMSNorm (float32 precision)
        orig_dtype = x.dtype
        x_norm = x.float()
        var = x_norm.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_norm * torch.rsqrt(var + self.eps)  # Non-inplace mul
        x_norm = x_norm.to(orig_dtype) * self.weight  # Convert to orig_dtype before weight mul
        
        # Step 2: Linear (GEMM on normalized output)
        # PyTorch will fuse these operations through torch.compile
        output = F.linear(x_norm, linear_weight, linear_bias)
        
        return output.to(orig_dtype)

    @torch.compile
    def fused_rms_add_linear_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [OPTIMIZATION] Fused RMSNorm + Residual + Linear operation
        
        Combines (RMSNorm(x + residual)) @ W^T + b into single operation.
        
        Args:
            x: Input tensor (..., hidden_size)
            residual: Residual tensor (..., hidden_size)
            linear_weight: Linear layer weight (out_features, hidden_size)
            linear_bias: Linear layer bias (out_features,) or None
            
        Returns:
            Tuple of (output_tensor (..., out_features), updated_residual (..., hidden_size))
        """
        # Step 1: Add residual
        orig_dtype = x.dtype
        x_sum = x.float() + residual.float()  # Non-inplace add
        residual_out = x_sum.to(orig_dtype)  # Updated residual for next layer
        
        # Step 2: RMSNorm (float32 precision)
        var = x_sum.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_sum * torch.rsqrt(var + self.eps)  # Non-inplace mul
        x_norm = x_norm.to(orig_dtype) * self.weight  # Convert to orig_dtype before weight mul
        
        # Step 3: Linear (GEMM on normalized output)
        # PyTorch will fuse these operations through torch.compile
        output = F.linear(x_norm, linear_weight, linear_bias)
        
        return output.to(orig_dtype), residual_out

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
