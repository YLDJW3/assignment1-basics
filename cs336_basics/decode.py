from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.transformer_lm import LM
from cs336_basics.utils import softmax, SPECIAL_TOKEN, log
import torch


def decoding(
    tokenizer: Tokenizer,
    model: LM,
    prompt: str,
    max_tokens: int,
    device: torch.device = None,
    top_p: float = 1,
    temperature: float = 1,
) -> str:
    x = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
    end_token_id = tokenizer.token_2_id[SPECIAL_TOKEN.encode()]
    for i in range(max_tokens):
        logits = model(x[-model.context_length :])
        if temperature != 1:
            logits = logits / temperature
        prob = softmax(logits[-1, :], -1)
        # top-p sampling
        if top_p < 1:
            prob = top_p_filter(prob, top_p)
        next_token = torch.multinomial(prob, num_samples=1)
        if next_token == end_token_id:
            break
        x = torch.concat([x, torch.tensor([next_token])], dim=-1)
        if i % 100 == 0:
            log.info(f"Decode tokens count {i + 1}")
    return tokenizer.decode(x.flatten().tolist())


def top_p_filter(prob: torch.Tensor, top_p: float) -> torch.Tensor:
    # filter top_p
    sorted_prob, sorted_index = torch.sort(prob, descending=True)
    cumsum = torch.cumsum(sorted_prob, dim=-1)
    mask = cumsum - sorted_prob > top_p
    sorted_prob[mask] = 0
    # re-normalize
    sorted_prob /= sorted_prob.sum(dim=-1, keepdim=True)
    # recover index
    prob = torch.zeros_like(prob)
    prob.scatter_(dim=-1, index=sorted_index, src=sorted_prob)
    return prob
