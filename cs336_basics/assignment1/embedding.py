import torch
from torch import nn


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        embedding_matrix = torch.zeros(
            [num_embeddings, embedding_dim], device=device, dtype=dtype
        )
        nn.init.trunc_normal_(embedding_matrix, mean=0, std=1, a=-3, b=3)
        self.embedding_model = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_model[token_ids]
