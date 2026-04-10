import torch
import torch.nn as nn
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.swiglu_feed_forward import SwigluFFN, SiluFFN
from cs336_basics.rms_normalization import RMSNorm
from cs336_basics.utils import log


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
        no_norm: bool = False,
        post_norm: bool = False,
        nope: bool = False,
        silu: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.no_norm = no_norm
        self.post_norm = post_norm
        self.nope = nope
        self.silu = silu
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        if silu:
            self.ffn = SiluFFN(
                d_model=d_model,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
            )
        else:
            self.ffn = SwigluFFN(
                d_model=d_model,
                d_ff=d_ff,
                device=device,
                dtype=dtype,
            )
        if not self.no_norm:
            self.rms_norm_attn = RMSNorm(d_model=d_model, device=device, dtype=dtype)
            self.rms_norm_ffn = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        token_positions = torch.arange(x.shape[-2])
        rope = not self.nope
        if self.no_norm:
            # no_norm
            x = x + self.attn(x, rope, token_positions)
            return x + self.ffn(x)
        elif self.post_norm:
            # post_norm
            x = self.rms_norm_attn(x + self.attn(x, rope, token_positions))
            return self.rms_norm_ffn(x + self.ffn(x))
        else:
            # default pre-norm
            x = x + self.attn(self.rms_norm_attn(x), rope, token_positions)
            return x + self.ffn(self.rms_norm_ffn(x))
