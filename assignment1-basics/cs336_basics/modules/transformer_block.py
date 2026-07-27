import torch
import torch.nn as nn


from cs336_basics.modules import RMSNorm, SwiGLU, MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()

        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

        self.attn = MultiheadAttention(d_model, num_heads, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # input should be [... sequence_length d_model]
        x = x + self.attn(self.ln1(x), torch.arange(x.size(-2)))

        return x + self.ffn(self.ln2(x))
