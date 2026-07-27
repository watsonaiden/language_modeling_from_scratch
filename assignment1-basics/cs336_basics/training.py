import math
from typing import Callable, Iterable, Optional

import torch


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
        Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
    """

    # softmax
    largest = torch.amax(inputs, dim=-1, keepdim=True)

    # subtracting out largest before exp keeps numbers on reasonable scale and prevents inf
    stable = inputs - largest

    log_sum = torch.logsumexp(stable, dim=-1, keepdim=True)

    return torch.mean(log_sum - torch.gather(stable, -1, targets.unsqueeze(-1)).squeeze(-1))


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr} (<0)")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            decay_step = lr * weight_decay

            for p in group["params"]:
                if p.grad is None:
                    continue
                # weight decay
                p.data -= decay_step * p.data

                grad = p.grad.data

                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)

                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad**2

                adjusted_lr = lr * math.sqrt(1 - (b2**t)) / (1 - b1**t)

                p.data -= adjusted_lr * m / (torch.sqrt(v) + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss


def lr_cosine_schedule(t, lr_max, lr_min, warmup_steps, annealing_steps):
    if t < warmup_steps:
        return t / warmup_steps * lr_max

    elif warmup_steps <= t <= annealing_steps:
        return lr_min + 0.5 * (1 + math.cos((t - warmup_steps) / (annealing_steps - warmup_steps) * math.pi)) * (
            lr_max - lr_min
        )

    # past annealing range
    else:
        return lr_min


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):

    rolling_l2 = 0

    for p in parameters:
        if p.grad is None:
            continue
        rolling_l2 += torch.square(p.grad).sum().item()

    rolling_l2 = math.sqrt(rolling_l2)

    if rolling_l2 >= max_l2_norm:
        scale_factor = max_l2_norm / (rolling_l2 + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(scale_factor)

    return parameters
