import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.utils import log


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        sin_vals, cos_vals = self.compute_rotation_matrix()
        self.register_buffer("sin_vals", sin_vals, persistent=False)
        self.register_buffer("cos_vals", cos_vals, persistent=False)

    def compute_rotation_matrix(self):
        # d_k/2 dimension
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2) / self.d_k))
        # max_seq_len dimension
        positions = torch.arange(self.max_seq_len)
        # (max_seq_len, d_k / 2)
        # theta_i_k where
        # i in [0, 1, ..., max_seq_len - 1]
        # k in [0, 2, ..., d_k - 2]
        angle = torch.outer(positions, inv_freq)
        log.debug(f"angle {angle.shape}")
        # (max_seq_len, d_k / 2)
        sin_vals = torch.sin(angle)
        cos_vals = torch.cos(angle)
        return sin_vals, cos_vals

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_k
        assert x.shape[-2] == token_positions.shape[-1]
        x_even = x[..., 0::2]  # (..., seq, d_k/2)
        x_odd = x[..., 1::2]
        sin_vals = self.sin_vals[token_positions]  # (seq, d_k/2)
        cos_vals = self.cos_vals[token_positions]
        log.debug(f"token_positions {token_positions.shape}")
        log.debug(f"x_even {x_even.shape}")
        log.debug(f"cos_vals {cos_vals.shape}")
        y_even = x_even * cos_vals - x_odd * sin_vals
        y_odd = x_even * sin_vals + x_odd * cos_vals
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)
