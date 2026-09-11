from einops import einsum
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    largest = torch.amax(x, dim=dim, keepdim=True)

    exp = torch.exp(x - largest)

    return exp / torch.sum(exp, dim=dim, keepdim=True)


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    # intutition, at index i,j this represents a raw score of how much segment i attends to segment j
    logits = einsum(Q, K, "... q_seq_len dk, ... k_seq_len dk -> ... q_seq_len k_seq_len") / (Q.size(-1) ** 0.5)

    # invert mask as we want to fill False values
    if mask is not None:
        logits = logits.masked_fill(~mask[: logits.size(-1), : logits.size(-1)], float("-inf"))

    return einsum(softmax(logits, -1), V, "... seq_len key, ... key dv -> ... seq_len dv")
