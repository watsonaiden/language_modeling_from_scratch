import torch
import torch.nn as nn


from einops import einsum, rearrange


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        angles = einsum(
            torch.arange(max_seq_len, device=device),
            1 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k)),
            "index, k -> index k",
        )

        self.position_sin = nn.Buffer(torch.sin(angles))
        self.position_cos = nn.Buffer(torch.cos(angles))

    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        """
        rotations are
        x1 = x*cos(theta) - y * sin(theta)
        y2 = x*sin(theta) + y * cos(theta)
        """
        ...

        sin = self.position_sin[token_position]  # num_tokens x d_k/2
        cos = self.position_cos[token_position]  # num_tokens x d_k/2

        evens = x[..., ::2]
        odds = x[..., 1::2]

        new_evens = cos * evens - sin * odds
        new_odds = sin * evens + cos * odds

        return rearrange([new_evens, new_odds], " pair ... d_k -> ... (d_k pair)")
