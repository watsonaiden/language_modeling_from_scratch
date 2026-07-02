import torch

import torch.nn as nn

from einops import rearrange

from cs336_basics.modules import Linear, RoPE

from .nn_utils import attention


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int):
        super().__init__()
        # represents 3 d_model x d_model layers in one
        # output dimensions represent each of the 3 fields (query, key, value) concated
        self.proj_qkv = Linear(d_model, d_model * 3)

        self.output_proj = Linear(d_model, d_model)

        self.rope = RoPE(theta=theta, d_k=d_model / num_heads, max_seq_len=max_seq_len)

        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        qkv = self.proj_qkv(x)

        qkv = rearrange(qkv, "... (three head d) -> three head ... d", three=3, head=self.num_heads)

        # token at i can only attend to tokens j <= i
        casual_mask = torch.tril(torch.ones((x.size(-2), x.size(-2)), dtype=torch.bool))

        rotated_query = self.rope(qkv[0], token_positions)
        rotated_key = self.rope(qkv[1], token_positions)

        out = attention(rotated_query, rotated_key, qkv[2], mask=casual_mask)

        heads_concat = rearrange(out, "heads ... d_k -> ... (heads d_k)")

        return self.output_proj(heads_concat)
