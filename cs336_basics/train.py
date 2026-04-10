import argparse
import math
import numpy as np
import time
import torch
import wandb
from collections.abc import Iterable
from cs336_basics.utils import *
from cs336_basics.transformer_lm import LM
from cs336_basics.adamw_optimizer import AdamW
from cs336_basics.loss_function import cross_entropy
from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.decode import decoding


def parse_args():
    p = argparse.ArgumentParser()
    # Mode
    p.add_argument("--mode", type=str, default="train", choices=["train", "valid", "decode"])
    p.add_argument("--name", type=str, default="")
    # Data
    p.add_argument("--train_data", type=str, required=None)
    p.add_argument("--valid_data", type=str, required=None)
    p.add_argument("--vocab_filepath", type=str, required=None)
    p.add_argument("--merge_filepath", type=str, required=None)
    p.add_argument("--checkpoint_dir", type=str, required=None)
    p.add_argument("--snapshot_filepath", type=str, default=None)
    # Model
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--theta", type=float, default=10000.0)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accumulate_batch_size", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--betas", type=tuple, default=(0.9, 0.95))
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--cp_interval", type=int, default=10)
    p.add_argument("--valid_interval", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_l2_norm", type=int, default=1e12)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr_max", type=float, default=0)
    p.add_argument("--lr_min", type=float, default=0)
    p.add_argument("--T_w", type=int, default=0)
    p.add_argument("--T_c", type=int, default=0)
    # Decoding
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--prompt", type=str)
    p.add_argument("--max_tokens", type=int, default=100_000)

    return p.parse_args()


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_tokens(path):
    tokens = np.load(path, mmap_mode="r")
    return tokens


def load_model(args, device, dtype):
    model = LM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        theta=args.theta,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        device=device,
        dtype=dtype,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {num_params:,}")
    # optimizer
    opt = AdamW(
        model.parameters(),
        args.lr,
        args.weight_decay,
        args.betas,
        args.eps,
        args.lr_max,
        args.lr_min,
        args.T_w,
        args.T_c,
    )
    start = 1
    # load checkpoint
    if args.snapshot_filepath is not None:
        start = load_checkpoint(args.snapshot_filepath, model, opt) + 1
        log.info(f"Load snapshot from {args.snapshot_filepath}, already trained {start - 1} steps")
    return model, opt, start


def gradient_clipping(t: int, parameters: Iterable[torch.nn.Parameter], maximum_l2_norm: float):
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
        log.info(f"step {t} gradient clipping, l2_norm {l2_norm} exceeds limit {maximum_l2_norm}")
        factor = maximum_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= factor


def train(args):
    # init
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    t0 = time.time()
    # wandb record
    name = args.name
    if args.name == "":
        name = f"d{args.d_model}_l{args.num_layers}_lr{args.lr}_batch{args.batch_size}_{int(t0)}"
    wandb.init(
        project="cs336-assignment1",
        name=name,
    )
    # data
    tokens = load_tokens(args.train_data)
    if args.valid_data is not None:
        valid_tokens = load_tokens(args.valid_data)
    # model
    model, opt, start = load_model(args, device, dtype)
    # model = torch.compile(model, backend="aot_eager")  # speed up training by JIT-compiling
    # training loop
    best_val_loss = float("inf")
    for t in range(start, args.max_steps + 1):
        x, y = data_loading(tokens, args.batch_size, args.context_length, device)
        # forward
        logits = model(x)
        loss = cross_entropy(logits, y)
        # backward
        loss = loss / args.accumulate_batch_size
        loss.backward()
        # only step the optimizer every `accumulate_batch_size`
        if t % args.accumulate_batch_size == 0:
            gradient_clipping(t, model.parameters(), args.max_l2_norm)
            opt.step()
            opt.zero_grad()
        # logging
        dt = time.time() - t0
        token_ps = t * args.batch_size * args.context_length / dt
        wandb.log(
            {
                "train/loss": loss.item(),
                "train/ppl": math.exp(loss.item()),
                "step": t,
            }
        )
        if t % args.log_interval == 0:
            log.info(f"step {t} | loss {loss.item():.4f} | ppl {math.exp(loss.item()):.2f} | {token_ps:,.0f} token/s")
        # checkpoint
        if t % args.cp_interval == 0:
            save_checkpoint(model, opt, t, args.checkpoint_dir + f"t{t}_checkpoint_{name}.pt")
        # validate
        if t % args.valid_interval == 0 and args.valid_data is not None:
            loss = valid(model, valid_tokens, t, args, device)
            if loss < best_val_loss:
                best_val_loss = loss
                save_checkpoint(model, opt, t, args.checkpoint_dir + f"best_model_t{t}_{name}.pt")
                log.info(f"save best model with val_loss {best_val_loss}")

    save_checkpoint(model, opt, args.max_steps, args.checkpoint_dir + f"final_model_{name}.pt")
    log.info("Training complete.")


def valid(model, valid_tokens, step, args, device):
    x, y = data_loading(valid_tokens, args.batch_size, args.context_length, device)
    logits = model(x)
    loss = cross_entropy(logits, y)
    log.info(f"step {step} | val_loss {loss.item():.4f} | val_ppl {math.exp(loss.item()):.2f}")
    if wandb.run is not None:
        wandb.log(
            {
                "val/loss": loss.item(),
                "val/ppl": math.exp(loss.item()),
                "step": step,
            }
        )
    return loss


if __name__ == "__main__":
    args = parse_args()
    log.info(f"args: {vars(args)}")
    # train
    if args.mode == "train":
        train(args)
    # validate using saved model
    elif args.mode == "valid":
        device = torch.device(args.device)
        dtype = DTYPE_MAP[args.dtype]
        model, _, step = load_model(args, device, dtype)
        valid_tokens = load_tokens(args.valid_data)
        _ = valid(model, valid_tokens, step, args, device)
    # decode
    elif args.mode == "decode":
        assert args.vocab_filepath is not None and args.merge_filepath is not None and args.prompt is not None
        device = torch.device(args.device)
        dtype = DTYPE_MAP[args.dtype]
        model, _, step = load_model(args, device, dtype)
        tokenizer = Tokenizer.from_files(
            Tokenizer,
            vocab_filepath=args.vocab_filepath,
            merges_filepath=args.merge_filepath,
            special_tokens=[SPECIAL_TOKEN],
        )
        y = decoding(tokenizer, model, args.prompt, args.max_tokens, 0.9, 1)
        log.info(f"Decode result: {y}")
