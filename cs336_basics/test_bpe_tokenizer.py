import pytest
from cs336_basics.bpe_tokenizer import *
from cs336_basics.test_bpe_train import SPECIAL_TOKEN


@pytest.mark.parametrize(
    "vocab_path, merge_path, special_tokens",
    [
        (
            "cs336_basics/TinyStoriesV2-GPT4-train-200M-vocab.json",
            "cs336_basics/TinyStoriesV2-GPT4-train-200M-merge.txt",
            [SPECIAL_TOKEN],
        ),
    ],
)
def test_from_files(vocab_path, merge_path, special_tokens):
    tokenizer = Tokenizer.from_files(Tokenizer, vocab_filepath=vocab_path, merges_filepath=merge_path, special_tokens=special_tokens)
    assert len(tokenizer.id_2_token) == 300
    assert len(tokenizer.token_2_id) == 300
    assert len(tokenizer.merges) == 43

@pytest.mark.parametrize(
    "tokenizer, text, expected",
    [
        (
            Tokenizer.from_files(Tokenizer, 
                                 vocab_filepath="cs336_basics/TinyStoriesV2-GPT4-train-200M-vocab.json", 
                                 merges_filepath="cs336_basics/TinyStoriesV2-GPT4-train-200M-merge.txt", 
                                 special_tokens=[SPECIAL_TOKEN],
                                 ),
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

@pytest.mark.parametrize(
    "tokenizer, tokens, expected",
    [
        (
            Tokenizer.from_files(Tokenizer, 
                                 vocab_filepath="cs336_basics/TinyStoriesV2-GPT4-train-200M-vocab.json", 
                                 merges_filepath="cs336_basics/TinyStoriesV2-GPT4-train-200M-merge.txt", 
                                 special_tokens=[SPECIAL_TOKEN],
                                 ),
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

@pytest.mark.parametrize(
    "tokenizer, texts",
    [
        (
            Tokenizer.from_files(Tokenizer, 
                                 vocab_filepath="cs336_basics/TinyStoriesV2-GPT4-train-200M-vocab.json", 
                                 merges_filepath="cs336_basics/TinyStoriesV2-GPT4-train-200M-merge.txt", 
                                 special_tokens=["<|endoffile|>", SPECIAL_TOKEN],
                                 ),
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
