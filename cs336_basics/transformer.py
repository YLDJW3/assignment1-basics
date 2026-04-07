import torch
import torch.nn as nn
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.swiglu_feed_forward import SwigluFFN
from cs336_basics.rms_normalization import RMSNorm


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwigluFFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.rms_norm_attn = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rms_norm_ffn = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        token_positions = torch.arange(x.shape[-2])
        x = x + self.attn(self.rms_norm_attn(x), True, token_positions)
        return x + self.ffn(self.rms_norm_ffn(x))
