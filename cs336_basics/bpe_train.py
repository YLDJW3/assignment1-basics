import os
import logging
import regex as re
from collections import Counter
from functools import partial
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CPU_NUM = 8

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    cpu_num: int = CPU_NUM,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1. Initialize vocabulary
    vocabulary = initalize_vocabulary(special_tokens)
    merges = []
    # 2. Pre-tokenize
    word_count = chunk_and_pre_tokenize(special_tokens, CPU_NUM, input_path)
    log.debug("pre-tokenize done")
    # 3. Generate initialize token pair count and index
    pair_count, pair_index = initialize_token_pair_count_and_index(word_count)
    log.debug(f"initialize token pair count and index done, got {len(pair_count)} token pairs")
    # 4. Iteratively count highest token pair, update pair_count and pair_index
    while len(vocabulary) < vocab_size:
        # count highest token pair
        highest_pair = get_highest_token_pair(pair_count)
        # update merges and vocabulary
        merges.append(highest_pair)
        vocabulary.append(highest_pair[0] + highest_pair[1])
        log.debug(f"{len(merges)} merge {merges[-1]}, vocab size is {len(vocabulary)}")
        # update pair_count and pair_index
        pair_count, pair_index = update_token_pair_count_and_index(highest_pair, pair_count, pair_index)
        log.debug(f"{len(merges)} update token pair count and index done, {len(pair_count)} token pairs")
    vocab = {}
    for i in range(len(vocabulary)):
        vocab[i] = vocabulary[i]
    return vocab, merges


def initalize_vocabulary(special_tokens: list[str]) -> list[bytes]:
    """
    initial vocabulary is the set of
        all bytes
        all special tokens
    """
    vocabulary = [bytes([i]) for i in range(256)]
    vocabulary.extend([token.encode("utf-8") for token in special_tokens])
    return vocabulary


def chunk_and_pre_tokenize(special_tokens: list[str], chunk_num: int, input_path: str | os.PathLike) -> list[str]:
    """
    split data into chunks by special tokens
    """
    # use first special token as chunk boundary
    first_speical_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_num, first_speical_token)
        log.debug(f"split into {len(boundaries) - 1} chunks")
        with Pool(chunk_num) as p:
            pre_token_counters = p.starmap(
                partial(pre_tokenize_by_offset, special_tokens, input_path), zip(boundaries[:-1], boundaries[1:])
            )
        return get_word_count(pre_token_counters)


def get_word_count(word_counters: list[Counter]) -> Counter:
    """
    different token pairs in identical words have the same effect
    dedup by word count improve training efficiency
    """
    word_count = Counter()
    for counter in word_counters:
        for word in counter:
            word_count[word] += counter[word]
    return word_count


def pre_tokenize_by_offset(special_tokens: list[str], input_path: str, start: int, end: int) -> Counter:
    """
    regex-based pre-tokenizer
    https://github.com/openai/tiktoken/pull/234/changes
    """
    spliter = "|".join([re.escape(special_token) for special_token in special_tokens])
    counter = Counter()
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        chunks = re.split(spliter, chunk)
        for chunk in chunks:
            for m in re.finditer(PAT, chunk):
                counter[m.group()] += 1
        return counter


def initialize_token_pair_count_and_index(word_count: Counter) -> tuple[Counter, dict[tuple[bytes, bytes], Counter]]:
    """
    count and index all token pairs, return
        pairs_count: counter of all token pairs
        pairs_index: index of all token pairs (set)
    """
    pairs_count = Counter()
    pairs_index = {}
    start = 0
    for word in word_count:
        b = word.encode("utf-8")
        word_bytes = len(b)
        for i in range(word_bytes - 1):
            pair = (b[i : i + 1], b[i + 1 : i + 2])
            if pair not in pairs_index:
                pairs_index[pair] = Counter()
            pairs_count[pair] += word_count[word]
            pairs_index[pair][start + i] += word_count[word]
        start += word_bytes
    return pairs_count, pairs_index


def get_highest_token_pair(pair_count: Counter) -> tuple[bytes, bytes]:
    """
    get token pair with highest frequency
    prefer the lexicographically greater pair when tie
    """
    highest_count = 0
    highest_pair = None
    for pair in pair_count:
        if pair_count[pair] > highest_count:
            highest_pair = pair
            highest_count = pair_count[pair]
        elif pair_count[pair] == highest_count and pair > highest_pair:
            highest_pair = pair
    return highest_pair


def update_token_pair_count_and_index(
    highest_pair: tuple[bytes, bytes],
    pair_count: Counter,
    pair_index: dict[tuple[bytes, bytes], Counter],
):
    """
    update token pair count and index after merge, return
        updated_pair_count
        updated_pair_index
    """
    updated_count = pair_count.copy()
    updated_index = pair_index.copy()
    for pair in pair_count:
        if pair == highest_pair:
            updated_count.pop(pair)
            updated_index.pop(pair)
        else:
            # two overlapping cases may be hit at the same time
            if pair[0] == highest_pair[1]:
                merged_pair = (highest_pair[0] + highest_pair[1], pair[1])
                highest_pair_first_token_bytes = len(highest_pair[0])
                # iterate pair instead of highest_pair first, reducing iterate times
                for idx in pair_index[pair].copy():
                    count = pair_index[pair][idx]
                    highest_pair_start = idx - highest_pair_first_token_bytes
                    if highest_pair_start in pair_index[highest_pair]:
                        if merged_pair not in updated_index:
                            updated_index[merged_pair] = Counter()
                        updated_count[merged_pair] += count
                        updated_index[merged_pair][highest_pair_start] = count
                        updated_count[pair] -= count
                        updated_index[pair].pop(idx)
            if pair[1] == highest_pair[0]:
                merged_pair = (pair[0], highest_pair[0] + highest_pair[1])
                pair_first_token_bytes = len(pair[0])
                for idx in pair_index[pair].copy():
                    count = pair_index[pair][idx]
                    if idx + pair_first_token_bytes in pair_index[highest_pair]:
                        if merged_pair not in updated_index:
                            updated_index[merged_pair] = Counter()
                        updated_count[merged_pair] += count
                        updated_index[merged_pair][idx] = count
                        updated_count[pair] -= count
                        updated_index[pair].pop(idx)
    return updated_count, updated_index
