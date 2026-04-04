from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay,
        betas,
        eps,
    ):
        assert len(betas) == 2
        defaults = {
            "lr": lr,
            "decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closrue: Optional[Callable] = None):
        loss = None if closrue is None else closrue()
        for group in self.param_groups:
            lr = group["lr"]
            decay = group["decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad.square()
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                p.data -= lr_t * m / torch.sqrt(v + eps)
                p.data -= lr * decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def learning_rate_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return lr_max * t / T_w
    elif t <= T_c:
        return lr_min + 1 / 2 * (lr_max - lr_min) * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi))
    else:
        return lr_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], maximum_l2_norm: float):
    eps = 1e-6
    l2_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.data
        l2_norm += grad.norm() ** 2
    l2_norm **= 0.5
    if l2_norm > maximum_l2_norm:
        # gradient clipping, scale grad down by factor max_l2_norm / (l2_norm + eps)
        factor = maximum_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= factor
