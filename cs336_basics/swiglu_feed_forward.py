import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.linear import Linear


class SwigluFFN(nn.Module):
    """
    d_ff should be approximately d_model * 8/3
    ensure the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        silu_w1x = self._silu(self.w1(x))
        w3x = self.w3(x)
        w1w3 = silu_w1x * w3x
        return self.w2(w1w3)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))
