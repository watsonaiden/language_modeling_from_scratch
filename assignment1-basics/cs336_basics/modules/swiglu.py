import torch

import torch.nn as nn


from . import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()

        # round up to nearest mult of 64
        # ~63 masks bottom 6 bits leaving only multiples of 64
        d_ff = d_ff or (8.0 * d_model / 3.0 + 63) & ~63

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1.forward(x)
        b = self.w3.forward(x)

        silu = a * torch.sigmoid(a)

        return self.w2.forward(silu * b)
