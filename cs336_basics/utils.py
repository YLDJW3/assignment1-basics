import os
import torch
import torch.nn as nn
import typing
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    e = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return e / e.sum(dim=dim, keepdim=True)


def data_loading(
    x: np.array,
    batch_size: int,
    context_length: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = np.random.randint(low=0, high=len(x) - context_length, size=(batch_size,))
    input = np.stack([x[s : s + context_length] for s in start])
    target = np.stack([x[s + 1 : s + 1 + context_length] for s in start])
    input = torch.from_numpy(input.astype(np.int64)).to(device)
    target = torch.from_numpy(target.astype(np.int64)).to(device)
    return input, target


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    """
    Dump all states of model, optimizer and iteration into file-like object
    """
    state = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer
):
    state = torch.load(src)
    assert "iteration" in state and "model" in state and "optimizer" in state
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]


def flops(layers: int, n: int, d_model: int, d_ff: int, vocab_size: int) -> int:
    output_embeddings = 2 * n * d_model * vocab_size
    ffn = layers * 6 * n * d_model * d_ff
    attn = layers * (4 * (n**2) * d_model + 6 * n * (d_model**2))
    flops = output_embeddings + ffn + attn
    log.info(f"output embedding costs {output_embeddings:,} FLOPs")
    log.info(f"ffn costs {ffn:,} FLOPs")
    log.info(f"attn costs {attn:,} FLOPs")
    log.info(f"total costs {flops:,} FLOPs")
    return flops


def parameter_num(layers: int, n: int, d_model: int, d_ff: int, vocab_size: int) -> int:
    token_embeddings = d_model * vocab_size
    output_embeddings = d_model * vocab_size
    attn = layers * 4 * d_model ^ 2
    ffn = layers * 3 * d_model * d_ff
    norm = layers * 2 * d_model
    post_norm = d_model
    total = token_embeddings + output_embeddings + attn + ffn + norm + post_norm
    log.info(f"output embedding has {output_embeddings:,} trainable parameters")
    log.info(f"token embedding has {token_embeddings:,} trainable parameters")
    log.info(f"ffn has {ffn:,} trainable parameters")
    log.info(f"attn has {attn:,} trainable parameters")
    log.info(f"norm has {norm:,} trainable parameters")
    log.info(f"post-norm has {post_norm:,} trainable parameters")
    log.info(f"total has {total:,} trainable parameters")
    return total


def memory_num(parameters: int, size: int) -> int:
    bytes = parameters * size
    log.info(f"{bytes}B={bytes // 1024}KB={bytes // (1024 * 1024)}MB={bytes // (1024 * 1024 * 1024)}GB is required")
    return bytes
