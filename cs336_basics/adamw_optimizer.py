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
        lr_max: float = 0,
        lr_min: float = 0,
        T_w: int = 0,
        T_c: int = 0,
    ):
        assert len(betas) == 2
        defaults = {
            "lr": lr,
            "lr_max": lr_max,
            "lr_min": lr_min,
            "T_w": T_w,
            "T_c": T_c,
            "decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closrue: Optional[Callable] = None):
        loss = None if closrue is None else closrue()
        for group in self.param_groups:
            decay = group["decay"]
            betas = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            lr_max = group["lr_max"]
            lr_min = group["lr_min"]
            T_w = group["T_w"]
            T_c = group["T_c"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                if lr_max != 0:
                    # enable lr scheduler
                    lr = learning_rate_schedule(t, lr_max, lr_min, T_w, T_c)
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
