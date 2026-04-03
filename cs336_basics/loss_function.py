import torch
import torch.nn as nn
from cs336_basics.utils import log


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    e = logits.exp()
    log_sum_e = e.sum(dim=-1).log()
    batch = targets.shape[-1]
    res = logits[torch.arange(batch), targets].squeeze(-1)
    log_p = res - log_sum_e
    entropy = -log_p.sum() / batch
    log.debug(f"entroy {entropy}")
    return entropy


def perplexity(entroy: torch.Tensor) -> torch.Tensor:
    return torch.exp(entroy.sum(dim=-1) / entroy.shape[-1])
