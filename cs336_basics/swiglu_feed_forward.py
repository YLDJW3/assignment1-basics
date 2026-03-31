import torch
import torch.nn as nn
from einops import einsum


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
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        silu_w1x = self._silu(einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff"))
        w3x = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        w1w3 = einsum(silu_w1x, w3x, "... d_ff, ... d_ff -> ... d_ff")
        return einsum(w1w3, self.w2, "... d_ff, d_model d_ff -> ... d_model")

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))
