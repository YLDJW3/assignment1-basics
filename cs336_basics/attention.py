import math
import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.utils import softmax, log
from cs336_basics.rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 100,
        theta: float = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

        self.q_weight = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.k_weight = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.v_weight = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.o_weight = nn.Parameter(torch.empty(self.d_model, self.d_model))

    def forward(
        self,
        x: torch.Tensor,
        rope: bool = False,
        token_positions: torch.Tensor | None = None,
    ):
        assert x.shape[-1] == self.d_model
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        Q = einsum(x, self.q_weight, "... seq_len d_model, h_d_k d_model -> ... seq_len h_d_k")
        K = einsum(x, self.k_weight, "... seq_len d_model, h_d_k d_model -> ... seq_len h_d_k")
        V = einsum(x, self.v_weight, "... seq_len d_model, h_d_v d_model -> ... seq_len h_d_v")
        # reshape to separate heads, (... seq_len h*d_k) -> (... h seq_len d_k)
        Q = Q.view(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*K.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*V.shape[:-1], self.num_heads, self.d_v).transpose(-3, -2)
        if rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        # (..., seq_len, d_model)
        multi_head_attention = scaled_dot_product_attention(Q, K, V, mask)
        # reshape (... h seq_len d_k) -> (... seq_len h*d_k)
        multi_head_attention = multi_head_attention.transpose(-3, -2)
        multi_head_attention = multi_head_attention.contiguous().view(
            *multi_head_attention.shape[:-2], self.num_heads * self.d_v
        )
        return einsum(multi_head_attention, self.o_weight, "... seq_len h_d_v, d_model h_d_v -> ... seq_len d_model")


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    assert K.shape[-1] == d_k
    qk = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if mask is not None:
        # mask map False to -inf
        qk = qk.masked_fill(~mask, float("-inf"))
    return einsum(softmax(qk, -1), V, "... n m, ... m d_v -> ... n d_v")
