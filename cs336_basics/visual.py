from cs336_basics.train import parse_args, load_model
from torchinfo import summary
import torch.nn as nn
import torch


def visualize(model: nn.Module, batch_size: int, context_length: int):
    summary(model, input_size=(batch_size, context_length), dtypes=[torch.int])


if __name__ == "__main__":
    args = parse_args()
    model, _, step = load_model(args, None, None)
    visualize(model, args.batch_size, args.context_length)
