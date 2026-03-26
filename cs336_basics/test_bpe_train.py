import pytest, cProfile
from cs336_basics.bpe_train import *


SPECIAL_TOKEN = "<|endoftext|>"


@pytest.mark.parametrize(
    "input_path, chunks_num",
    [
        ("data/test_get_chunks_1.txt", 1),
        # ("data/test_get_chunks_2.txt", 2),
        ("data/TinyStoriesV2-GPT4-valid.txt", 12),
    ],
)
def test_get_chunks(input_path, chunks_num):
    chunks = get_chunks([SPECIAL_TOKEN], chunks_num, input_path)
    assert len(chunks) == chunks_num


@pytest.mark.parametrize(
    "special_token, vocab_size",
    [
        ([SPECIAL_TOKEN], 257),
    ],
)
def test_initalize_vocabulary(special_token, vocab_size):
    vocab = initalize_vocabulary(special_token)
    assert len(vocab) == vocab_size


@pytest.mark.parametrize(
    "input_path, expected_len",
    [
        ("data/test_get_chunks_1.txt", 209),
        # ("data/test_get_chunks_2.txt", 1131),
    ],
)
def test_pre_tokenize(input_path, expected_len):
    chunks = get_chunks([SPECIAL_TOKEN], 1, input_path)
    assert len(chunks) == 1
    tokens = pre_tokenize(chunks[0])
    assert len(tokens) == expected_len


@pytest.mark.parametrize(
    "input_path",
    [
        ("data/test_get_chunks_1.txt"),
        # ("data/test_get_chunks_2.txt"),
    ],
)
def test_initialize_token_pair_count_and_index(input_path):
    chunks = get_chunks([SPECIAL_TOKEN], 1, input_path)
    assert len(chunks) == 1
    pre_tokens = pre_tokenize(chunks[0])
    word_count = get_word_count([pre_tokens])
    pair_count, pair_index = initialize_token_pair_count_and_index(word_count)
    assert len(pair_count) > 0
    assert len(pair_count) == len(pair_index)
    # validate pair_index includes all token pairs
    index_size = 0
    for word in word_count:
        b = word.encode("utf-8")
        index_size += len(b) - 1
    index_set = set()
    for s in pair_index.values():
        index_set = index_set.union(s.keys())
    assert index_size == len(index_set)


@pytest.mark.parametrize(
    "input_path, pair_count, expected",
    [
        ("data/test_get_chunks_1.txt", Counter(), (bytes([32]), bytes([116]))),
        # ("data/test_get_chunks_2.txt", Counter(), (bytes([32]), bytes([116]))),
        (
            "",
            Counter({(bytes([32]), bytes([116])): 100, (bytes([100]), bytes([116])): 50}),
            (bytes([32]), bytes([116])),
        ),
        (
            "",
            Counter({(bytes([32]), bytes([116])): 100, (bytes([100]), bytes([116])): 100}),
            (bytes([100]), bytes([116])),
        ),
    ],
)
def test_get_highest_token_pair(input_path, pair_count, expected):
    if input_path != "":
        chunks = get_chunks([SPECIAL_TOKEN], 1, input_path)
        assert len(chunks) == 1
        pre_tokens = pre_tokenize(chunks[0])
        word_count = get_word_count([pre_tokens])
        pair_count, _ = initialize_token_pair_count_and_index(word_count)
    highest_pair = get_highest_token_pair(pair_count)
    assert highest_pair == expected


@pytest.mark.parametrize(
    "input_path, expected_highest_pair",
    [
        ("data/test_get_chunks_1.txt", (bytes([32]), bytes([116]))),
        # ("data/test_get_chunks_2.txt", (bytes([32]), bytes([116]))),
    ],
)
def test_update_token_pair_count_and_index(input_path, expected_highest_pair):
    chunks = get_chunks([SPECIAL_TOKEN], 1, input_path)
    assert len(chunks) == 1
    pre_tokens = pre_tokenize(chunks[0])
    word_count = get_word_count([pre_tokens])
    pair_count, pair_index = initialize_token_pair_count_and_index(word_count)
    highest_pair = get_highest_token_pair(pair_count)
    assert highest_pair == expected_highest_pair
    updated_count, updated_index = update_token_pair_count_and_index(highest_pair, pair_count, pair_index)
    assert updated_count.keys() == updated_index.keys()

@pytest.mark.parametrize(
    "pair_count, pair_index, expected_highest_pair, expected_updated_count, expected_updated_index",
    [
        (
            # 32 116 100 44
            # 32 116
            # 116 100 44
            Counter({(bytes([32]), bytes([116])): 2, (bytes([116]), bytes([100])): 2, (bytes([100]), bytes([44])): 2}),

            {(bytes([32]), bytes([116])): Counter({0: 1, 4: 1}), (bytes([116]), bytes([100])): Counter({1: 1, 6: 1}), (bytes([100]), bytes([44])): Counter({2: 1, 7: 1})},

            (bytes([116]), bytes([100])),

            Counter({(bytes([32]), bytes([116])): 1, (bytes([32]), bytes([116, 100])): 1, (bytes([116, 100]), bytes([44])): 2, (bytes([100]), bytes([44])): 0}),

            {(bytes([32]), bytes([116])): Counter({0: 0, 4: 1}), (bytes([32]), bytes([116, 100])): Counter({0: 1}), (bytes([116, 100]), bytes([44])): Counter({1: 1, 6: 1}), (bytes([100]), bytes([44])): Counter()},
        ),
        (
            # 32 116 100 44
            # 32 116
            # 116 100 44
            Counter({(bytes([32]), bytes([116])): 1, (bytes([32]), bytes([116, 100])): 1, (bytes([116, 100]), bytes([44])): 2}),

            {(bytes([32]), bytes([116])): Counter({4: 1}), (bytes([32]), bytes([116, 100])): Counter({0: 1}), (bytes([116, 100]), bytes([44])): Counter({1: 1, 6: 1})},

            (bytes([116, 100]), bytes([44])),

            Counter({(bytes([32]), bytes([116])): 1, (bytes([32]), bytes([116, 100])): 0, (bytes([32]), bytes([116, 100, 44])): 1}),

            {(bytes([32]), bytes([116])): Counter({4: 1}), (bytes([32]), bytes([116, 100])): Counter({0: 0}), (bytes([32]), bytes([116, 100, 44])): Counter({0: 1})},
        ),
    ],
)
def test_update_token_pair_count_and_index_raw(
    pair_count, pair_index, expected_highest_pair, expected_updated_count, expected_updated_index
):
    highest_pair = get_highest_token_pair(pair_count)
    assert highest_pair == expected_highest_pair
    updated_count, updated_index = update_token_pair_count_and_index(highest_pair, pair_count, pair_index)
    assert updated_count == expected_updated_count
    assert updated_index == expected_updated_index


@pytest.mark.parametrize(
    "input_path, vocab_size, special_tokens, expected_merge",
    [
        ("data/test_get_chunks_1.txt", 258, [SPECIAL_TOKEN], [(bytes([32]), bytes([116]))]),
        # ("data/test_get_chunks_2.txt", 300, [SPECIAL_TOKEN], []),
        ("data/TinyStoriesV2-GPT4-valid.txt", 300, [SPECIAL_TOKEN], []),
        # ("data/TinyStoriesV2-GPT4-train-200M.txt", 300, [SPECIAL_TOKEN], []),
    ],
)
def test_train_bpe(input_path, vocab_size, special_tokens, expected_merge):
    # profiler = cProfile.Profile()
    # profiler.enable()
    vocab, merge = train_bpe(input_path, vocab_size, special_tokens)
    # profiler.disable()
    # profiler.print_stats(sort='tottime')

    assert len(vocab) == vocab_size
    if len(expected_merge) > 0:
        for i in range(len(expected_merge)):
            assert vocab[257 + i] == expected_merge[i][0] + expected_merge[i][1]
        assert merge == expected_merge
    else:
        print(vocab)
        print(merge)
