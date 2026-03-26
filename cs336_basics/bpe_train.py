import os, json
import logging
import regex as re
from collections.abc import Iterator
from collections import Counter
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
    # 2. Split the file into chunks by special token
    chunks = get_chunks(special_tokens, CPU_NUM, input_path)
    log.debug(f"split into {len(chunks)} chunks")
    # 3. Parallel pre-tokenize
    with Pool(CPU_NUM) as p:
        pre_token_chunks = p.map(pre_tokenize, chunks)
    log.debug(f"pre-tokenize done, got {len(pre_token_chunks)} pre token chunks")
    # 4. Generate initialize token pair count and index
    word_count = get_word_count(pre_token_chunks)
    pair_count, pair_index = initialize_token_pair_count_and_index(word_count)
    log.debug(f"initialize token pair count and index done, got {len(pair_count)} token pairs")
    # 5. Iteratively count highest token pair, update pair_count and pair_index
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

def get_word_count(word_chunks: list[list[str]]) -> Counter:
    """
    different token pairs in identical words have the same effect 
    dedup by word count improve training efficiency
    """
    word_count = Counter()
    for chunk in word_chunks:
        for word in chunk:
            word_count[word] += 1
    return word_count
    

def initalize_vocabulary(special_tokens: list[str]) -> list[bytes]:
    """
    initial vocabulary is the set of
        all bytes
        all special tokens
    """
    vocabulary = [bytes([i]) for i in range(256)]
    vocabulary.extend([token.encode("utf-8") for token in special_tokens])
    return vocabulary


def get_chunks(special_tokens: list[str], chunk_num: int, input_path: str | os.PathLike) -> list[str]:
    """
    split data into chunks, splited by special tokens
    """
    pattern = "|".join([re.escape(special_token) for special_token in special_tokens])
    with open(input_path, "rb") as f:
        # use first special token as chunk boundary
        first_speical_token = special_tokens[0].encode('utf-8')
        boundaries = find_chunk_boundaries(f, chunk_num, first_speical_token)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append("".join(re.split(pattern, chunk)))
        return chunks


def pre_tokenize(data: str, pattern: str = PAT) -> list[str]:
    """
    regex-based pre-tokenizer
    https://github.com/openai/tiktoken/pull/234/changes
    """
    return [m.group() for m in re.finditer(pattern, data)]


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
    with open('cs336_basics/pairs_index.json', 'w') as f:
        json.dump({str(k): v for k, v in pairs_index.items()}, f, indent=4)
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
