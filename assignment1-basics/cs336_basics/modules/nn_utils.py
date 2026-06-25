from einops import einsum
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    largest = torch.amax(x, dim=dim, keepdim=True)

    exp = torch.exp(x - largest)

    return exp / torch.sum(exp, dim=dim, keepdim=True)


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    logits = einsum(Q, K, "... q dk, ... k dk -> ... q k") / (Q.size(-1) ** 0.5)

    # invert mask as we want to fill False values
    if mask:
        logits = logits.masked_fill(~mask, float("-inf"))

    return einsum(softmax(logits, -1), V, "... q k, ... k dv -> ... q dv")
