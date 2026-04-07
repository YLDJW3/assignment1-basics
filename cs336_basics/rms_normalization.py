import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm(x) = gamma * x / sqrt(mean(x ^ 2) + epsilon)
        """
        assert x.shape[-1] == self.d_model
        in_dtype = x.dtype
        # upcast dtype to float32 to avoid square overflow
        x = x.to(torch.float32)
        rms = torch.sqrt(x.square().mean(dim=-1) + self.eps)
        x_norm = x / rms.unsqueeze(-1)
        result = einsum(x_norm, self.weight, "... d_model, d_model -> ... d_model")
        # downcast to input dtype
        return result.to(in_dtype)
