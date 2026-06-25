import torch
import torch.nn as nn

from jaxtyping import Shaped


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.embeddings, std=1, a=-3, b=3)

    def forward(self, token_ids: Shaped[torch.Tensor, "token_d"]) -> Shaped[torch.Tensor, "token_d embedding_d"]:
        return self.embeddings[token_ids]
