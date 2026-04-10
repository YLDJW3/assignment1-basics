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
        norm_experiment: bool=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.post_norm = RMSNorm(d_model, device=device, dtype=dtype)
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
                    norm_experiment=norm_experiment,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.post_norm(x)
        x = self.linear(x)
        return x
