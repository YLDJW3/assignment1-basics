import torch
import torch.nn as nn
from cs336_basics.utils import log


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, context_length, vocab_size)
    targets: (batch_size, context_length)
    """
    logits = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_e = logits.exp().sum(dim=-1).log()
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).unsqueeze(-1)
    log_p = target_logits - log_sum_e
    entropy = -log_p.mean()
    log.debug(f"entroy {entropy}")
    return entropy


def perplexity(entroy: torch.Tensor) -> torch.Tensor:
    return torch.exp(entroy.sum(dim=-1) / entroy.shape[-1])
