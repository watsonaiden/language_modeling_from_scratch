import torch
import torch.nn as nn


from einops import reduce, rearrange


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        rms = torch.sqrt((reduce(x**2, "... d -> ...", "mean") + self.eps))
        rms = rearrange(rms, "... -> ... 1")

        return (x / rms) * self.weight
