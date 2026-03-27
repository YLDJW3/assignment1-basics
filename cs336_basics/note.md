# Overview
1. What you will implement
    1. Byte-pair encoding (BPE) tokenizer (§2)
    2. Transformer language model (LM) (§3)
    3. The cross-entropy loss function and the AdamW optimizer (§4)
    4. The training loop, with support for serializing and loading model and optimizer state (§5)
2. What you will run
    1. Train a BPE tokenizer on the TinyStories dataset.
    2. Run your trained tokenizer on the dataset to convert it into a sequence of integer IDs.
    3. Train a Transformer LM on the TinyStories dataset.
    4. Generate samples and evaluate perplexity using the trained Transformer LM.
    5. Train models on OpenWebText and submit your attained perplexities to a leaderboard.
3. Data requirement
    1. TinyStories dataset
    2. OpenWebText

# BPE tokenizer
1. Subword tokenizer: tradeoff between **vocabulary size** and **compression ratio**
    Word-level tokenizer: large vocabulary size
    Byte-level tokenizer: limited vocabulary size, but low compression ratio
2. BPE: a compression algorithm that iteratively replaces (merges) the **most frequent pair of bytes** with a single, new unused index

## BPE traning
1. Steps
    1. Vocabulary initialization: initial vocabulary is the set of all bytes (size 256)
    2. Pre-tokenize: a coarse-grained tokenization over the corpus that helps us count how often pairs of characters appear
        - original BPE implementation: split on whitespace
        - regex-based pre-tokenizer used by GPT-2 https://github.com/openai/tiktoken/pull/234/changes
        ```python
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ```
    3. Compute BPE merges
        - Iteratively counts every pair of bytes and identifies the **pair with the highest frequency**
        - Add new merged token to the vocabulary
        - Not consider pairs that cross pre-token boundaries
        - Deterministically break ties in pair frequency by preferring the **lexicographically greater pair** 
    4. Speical tokens
        -  Some strings (e.g., `<|endoftext|>`) are used to encode **metadata** (e.g., boundaries between documents), and should be **preserved as a single token**
        - Initialize vocabulary with 256 bytes and all speical tokens
2. Experiment
    1. **Parallelizing** pre-tokenization
        - chunk the corpus while ensuring your **chunk boundaries** occur at the beginning of a special token, refer to `pretokenization_example.py`
        - parallelizing your code with the built-in library `multiprocessing`
    2. Remove special tokens before pre-tokenization
    3. Optimizing the merging step
        - the only pair counts that change after each merge are those that **overlap with the merged pair**
        - **indexing** the counts of all pairs and incrementally updating these counts
3. Tips
    1. use profiling tools like `cProfile` or `scalene` to identify the bottlenecks in your implementation
    2. first train on a **small debug dataset** to speed up development

## Encoding and Decoding
1. Encoding text: a BPE tokenizer vocabulary and a list of BPE merges
    1. Pre-tokenize with the same rule
    2. Apply the merges in the same order of creation in training
    3. Memory considerations: break large files up into manageable chunks and process each chunk in-turn
2. Decoding text
    1. look up each ID’s corresponding entries in the vocabulary
    2. when input token IDs do not produce a valid Unicode string, replace the malformed bytes with the official Unicode replacement character `U+FFFD`

# Answers
## Problem (unicode1): Understanding Unicode (1 point)
1. What Unicode character does chr(0) return?
    '\x00'
2. How does this character’s string representation (__repr__()) differ from its printed representation?
    "'\\x00'"
3. What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
```python
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```
## Problem (unicode2): Unicode Encodings (3 points)
1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
    UTF-16 and UTF-32 have much more null or padding bytes than UTF-8
2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
    A character could be decoded into 1-4 bytes in UTF-8, for those characters decoded into 2-4 bytes, the function is incorrect, like "你好"
3. Give a two byte sequence that does not decode to any Unicode character(s).
    1. `b'\xc0\x80'.decode('utf-8')` invalid start byte 
    2. `b'\xe4\xbd'.decode('utf-8')` unexpected end of data
    3. A two-byte unicode character requires the first byte to be `110xxxxx` and the second to be `10xxxxxx`, and each code point must use the **shortest byte sequence** that can represent it
## Problem (train_bpe): BPE Tokenizer Training (15 points)
1. Optional optimization (large time-investment)
    - Implement the key parts of your training method using some **systems language**, for instance C++ (consider cppyy for this) or Rust (using PyO3)
    - Be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. 
    - Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex package in Python is, if anything, even faster.
### Implementation
1. Implement in cs_336_basics/bpe_train.py
2. Optimization  
    1. Parallelize: chunk data file and parallelize pre-tokenization with multiprocessing (CPU)
    2. Pre-token dedup: identical pre-tokens have the same effect on BPE training, store as a pre-token counter (CPU, Memory) 
    3. `re.finditer`: use `re.finditer` instead of `re.findAll`, store in counter can avoid storing all pre-tokens (Memory)
    4. iteration times: `update_token_pair_count_and_index` iterate current pair's index (instead of highest pair), reducint the iteration times (CPU)

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
1. `TinyStoriesV2-GPT4-train-10_000vocab-8C.log`, traning takes 97s
2. `TinyStoriesV2-GPT4-train-10_000vocab-8C-vocab.json`
3. `TinyStoriesV2-GPT4-train-10_000vocab-8C-merge.txt`
## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
TODO
## Problem (tokenizer): Implementing the tokenizer (15 points)
> implement in cs_336_basics/bpe_tokenizer.py
1. Implement a `Tokenizer` class that, given a `vocabulary` and a list of `merges`, `encodes`
text into integer IDs and `decodes` integer IDs into text. 
2. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there). We recommend the following interface
3. Test
    implement the test adapter at [adapters.get_tokenizer]
    run `uv run pytest tests/test_tokenizer.py`
## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
1. Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s **compression ratio** (bytes/token)
    TODO
2. What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens
    TODO
3. Estimate the throughput of your tokenizer (e.g., in **bytes/second**). How long would it take to
tokenize the Pile dataset (825GB of text)?
    TODO
4. Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a `NumPy` array of datatype `uint16`. Why is `uint16` an appropriate choice?
    Vocabulary size is larger than 256, but much smaller than 65536