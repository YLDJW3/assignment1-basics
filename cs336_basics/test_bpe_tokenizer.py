import pytest
import numpy as np
from pathlib import Path
from cs336_basics.bpe_tokenizer import *
from cs336_basics.test_bpe_train import SPECIAL_TOKEN

VOCAB_PATH_200MB = "cs336_basics/TinyStoriesV2-GPT4-train-200M-vocab.json"
MERGE_PATH_200MB = "cs336_basics/TinyStoriesV2-GPT4-train-200M-merge.txt"


@pytest.fixture
def tokenizer_200MB():
    return Tokenizer.from_files(
        Tokenizer,
        vocab_filepath=VOCAB_PATH_200MB,
        merges_filepath=MERGE_PATH_200MB,
        special_tokens=[SPECIAL_TOKEN],
    )


@pytest.mark.skipif(not Path(VOCAB_PATH_200MB).exists(), reason="test data file missed")
def test_from_files():
    tokenizer = tokenizer_200MB
    assert len(tokenizer.id_2_token) == 300
    assert len(tokenizer.token_2_id) == 300
    assert len(tokenizer.merges) == 43


@pytest.mark.skipif(not Path(VOCAB_PATH_200MB).exists(), reason="test data file missed")
@pytest.mark.parametrize(
    "tokenizer, text, expected",
    [
        (
            tokenizer_200MB,
            "hello",
            [258, 294, 111],
        ),
    ],
)
def test_encode(tokenizer, text, expected):
    got = tokenizer.encode(text)
    if len(expected) > 0:
        assert got == expected
    else:
        print(got)


@pytest.mark.skipif(not Path(VOCAB_PATH_200MB).exists(), reason="test data file missed")
@pytest.mark.parametrize(
    "tokenizer, tokens, expected",
    [
        (
            tokenizer_200MB,
            [258, 294, 111],
            "hello",
        ),
    ],
)
def test_decode(tokenizer: Tokenizer, tokens: list[int], expected: str):
    got = tokenizer.decode(tokens)
    if len(expected) > 0:
        assert got == expected
    else:
        print(got)


@pytest.mark.skipif(not Path(VOCAB_PATH_200MB).exists(), reason="test data file missed")
@pytest.mark.parametrize(
    "tokenizer, texts",
    [
        (
            tokenizer_200MB,
            [
                # "hello world",
                # "code is cheap, show me the talk",
                # "中国",
                # "✅❌",
                "Héllò hôw <|endoftext|><|endoffile|> are ü? 🙃<|endoffile|>",
            ],
        ),
    ],
)
def test_encode_decode_roundtrip(tokenizer: Tokenizer, texts: list[str]):
    for text in texts:
        tokens = tokenizer.encode(text)
        got = tokenizer.decode(tokens)
        assert got == text


@pytest.mark.parametrize(
    "vocab_filepath, merges_filepath, special_tokens, input_filepath, output_filepath, token_num",
    [
        # (
        #     "cs336_basics/TinyStoriesV2-GPT4-train-10_000vocab-8C-vocab.json",
        #     "cs336_basics/TinyStoriesV2-GPT4-train-10_000vocab-8C-merge.txt",
        #     [SPECIAL_TOKEN],
        #     "data/TinyStoriesV2-GPT4-valid.txt",
        #     "data/TinyStoriesV2-GPT4-valid-tokens.npy",
        #     5469572,
        # ),
        (
            "cs336_basics/TinyStoriesV2-GPT4-train-10_000vocab-8C-vocab.json",
            "cs336_basics/TinyStoriesV2-GPT4-train-10_000vocab-8C-merge.txt",
            [SPECIAL_TOKEN],
            "data/TinyStoriesV2-GPT4-train.txt",
            "data/TinyStoriesV2-GPT4-train-tokens.npy",
            0,
        )
    ],
)
def test_encode_and_save_np_array(
    vocab_filepath, merges_filepath, special_tokens, input_filepath, output_filepath, token_num
):
    tokenizer = Tokenizer.from_files(Tokenizer, vocab_filepath, merges_filepath, special_tokens)
    chunk_size = 64 * 1024 * 1024
    max_tokens = 600_000_000
    buffer = ""
    offset = 0
    bin_path = output_filepath.replace(".npy", ".bin")
    fp = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(max_tokens,))
    with open(input_filepath) as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                if buffer:
                    token_ids = tokenizer.encode(buffer)
                    fp[offset : offset + len(token_ids)] = token_ids
                    offset += len(token_ids)
                break
            buffer += data
            last_nl = buffer.rfind("\n")
            if last_nl != -1:
                token_ids = tokenizer.encode(buffer[: last_nl + 1])
                fp[offset : offset + len(token_ids)] = token_ids
                offset += len(token_ids)
                buffer = buffer[last_nl + 1 :]
    fp.flush()
    final = np.memmap(bin_path, dtype=np.uint16, mode="r+", shape=(offset,))
    final.flush()
    np.save(output_filepath, np.array(final))

    if token_num > 0:
        assert offset == token_num
    else:
        log.info(f"token num {offset}")
