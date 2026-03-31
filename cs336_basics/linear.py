import math
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.weight = nn.Parameter(torch.zeros(self.d_out, self.d_in, dtype=dtype, device=device))
        std = math.sqrt(2 / (self.d_in + self.d_out))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x @ self.weight.T
        """
        assert x.shape[-1] == self.d_in
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
