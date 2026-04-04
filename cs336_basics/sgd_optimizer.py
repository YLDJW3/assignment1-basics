from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closrue: Optional[Callable] = None):
        loss = None if closrue is None else closrue()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # get state associated with parameter p
                t = state.get("t", 0)  # get iteration number from the state
                grad = p.grad.data  # get gradient of loss with respect to p
                p.data -= lr / math.sqrt(t + 1) * grad  # update weight tensor in place
                state["t"] = t + 1  # increase iteration number
        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad()  # reset gradientes
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()  # run backward pass, computes gradients
        opt.step()  # run optimizer step
