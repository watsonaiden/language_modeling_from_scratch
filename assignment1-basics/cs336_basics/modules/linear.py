import torch

import torch.nn as nn

from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))

        std = (2 / (in_features + out_features)) ** (1 / 2)
        nn.init.trunc_normal_(self.weights, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... i, o i -> ... o")
