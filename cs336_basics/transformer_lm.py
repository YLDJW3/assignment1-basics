import torch
import torch.nn as nn
from cs336_basics.rms_normalization import RMSNorm
from cs336_basics.transformer import Transformer
from cs336_basics.utils import softmax, log
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding


class LM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device=None,
        dtype=None,
        no_norm: bool = False,
        post_norm: bool = False,
        nope: bool = False,
        silu: bool = False,
    ):
        super().__init__()
        # hyper parameter
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.theta = theta
        # experiment
        self.no_norm = no_norm
        self.post_norm = post_norm
        self.nope = nope
        self.silu = silu

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.post_norm_layer = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.transformers = nn.Sequential(
            *[
                Transformer(
                    d_model,
                    num_heads,
                    context_length,
                    theta,
                    d_ff,
                    device=device,
                    dtype=dtype,
                    no_norm=no_norm,
                    post_norm=post_norm,
                    nope=nope,
                    silu=silu,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.post_norm_layer(x)
        x = self.linear(x)
        return x
