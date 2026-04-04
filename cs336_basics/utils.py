import torch
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
    tokens = torch.from_numpy(x)
    assert batch_size * context_length + 1 <= len(tokens)
    start = range(0, batch_size * context_length, context_length)
    input = torch.stack([tokens[s: s + context_length] for s in start])
    target = torch.stack([tokens[s + 1: s + 1 + context_length] for s in start])
    return input, target



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
