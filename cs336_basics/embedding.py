import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_matrix, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        table lookup on embedding_matrix
        """
        assert (token_ids < self.vocab_size).all()
        return self.embedding_matrix[token_ids]
