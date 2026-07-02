import torch
import torch.nn as nn

from cs336_basics.modules import TransformerBlock, Embedding, RMSNorm, Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)]
        )

        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embeddings(x)

        output = self.layers(embeddings)

        return self.lm_head(self.ln_final(output))
