import pytest
from cs336_basics.utils import flops, log, parameter_num, memory_num

@pytest.mark.parametrize(
    "name,layers,n,d_model,d_ff,vocab_size,expected",
    [
        (
            "GPT-2 XL",
            48, 
            1024,
            1600,
            6400,
            50_257,
            4_261_678_284_800,
        ),
        (
            "GPT-2 XL context-length",
            48, 
            16_384,
            1600,
            6400,
            50_257,
            0,
        )
    ]
)
def test_flops(name: str, layers: int, n: int, d_model: int, d_ff: int, vocab_size: int, expected: int):
    log.info(f"{name} case")
    got = flops(layers, n, d_model, d_ff, vocab_size)
    if expected > 0:
        assert expected == got


@pytest.mark.parametrize(
    "name,layers,n,d_model,d_ff,vocab_size,expected",
    [
        (
            "GPT-2 XL",
            48, 
            1024,
            1600,
            6400,
            50_257,
            0,
        ),
    ]
)
def test_parameter_num(name: str, layers: int, n: int, d_model: int, d_ff: int, vocab_size: int, expected: int):
    log.info(f"{name} case")
    got = parameter_num(layers, n, d_model, d_ff, vocab_size)
    _ = memory_num(got, 4)
    if expected > 0:
        assert expected == got

