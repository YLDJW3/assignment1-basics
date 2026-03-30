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

## Answers
### Problem (unicode1): Understanding Unicode (1 point)✅
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
### Problem (unicode2): Unicode Encodings (3 points)✅
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
### Problem (train_bpe): BPE Tokenizer Training (15 points)✅
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

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)✅
1. `TinyStoriesV2-GPT4-train-10_000vocab-8C.log`, traning takes 97s
2. `TinyStoriesV2-GPT4-train-10_000vocab-8C-vocab.json`
3. `TinyStoriesV2-GPT4-train-10_000vocab-8C-merge.txt`
### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)⭕
TODO
### Problem (tokenizer): Implementing the tokenizer (15 points)✅
1. Implement a `Tokenizer` class that, given a `vocabulary` and a list of `merges`, `encodes`
text into integer IDs and `decodes` integer IDs into text. 
2. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there). We recommend the following interface
3. Test
    implement the test adapter at [adapters.get_tokenizer]
    run `uv run pytest tests/test_tokenizer.py`
### Implementation
1. Implement in `cs_336_basics/bpe_tokenizer.py`
2. Test results in `tests/output_test_tokenizer.txt`

### Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
1. Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s **compression ratio** (bytes/token)
    TODO
2. What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens
    Guess: compression ratio worse than using according tokenizer?
3. Estimate the throughput of your tokenizer (e.g., in **bytes/second**). How long would it take to tokenize the Pile dataset (825GB of text)?
    TODO
4. Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a `NumPy` array of datatype `uint16`. Why is `uint16` an appropriate choice?✅
    Vocabulary size is larger than 256, but much smaller than 65536
### Implementation
1. Encoding `TinyStoriesV2-GPT4-valid.txt` into np array takes **139.78s** on a 18C mac, implying under-optimization
2. Optimization
    1. Set lru_cache maxsize to None, running time decreases to **23.27s**
    2. Parallel pre-tokenization and apply merge in main process, running time decreases to **17.08s** (`multiprocessing.Pool` do not share lru_cache)
3. Merge process optimization
    1. After optimization, encoding `TinyStoriesV2-GPT4-train.txt` still takes 250s, and applying priority-based merge can reduce it to 148s
    2. **Naive merge**: iterate every merge in order and compare the merge to every token pair in the word, time complexity is `O(word_size * merge_count)`
    3. **Priority-based merge**: iterate every token pair in the word to find the rank-first merge, time complexity is `O(word_size^2 * applied_merge_count)`. If word_size * applied_merge_count < merge_count, this appoach is faster than naive approach.

# Transformer model
## Encoder: self-attention + feed forward
1. Self-attention layer
    1. Calculate query, key, value vectors for embedding vectors of words
    2. Calculate scores by taking the dot product of query vector of current word with key vector of each word
        When processing self-attention of word i, `socre_i_j = query_i * key_j` for every word j
    3. Divide the scores by the squre root of dimension of they key vecotrs (by default is 8)
    4. Softmax score determines how much each word will be expressed at this position
    5. Multiply each value vactor by the softmax score to get weighted value vector
    6. Sum up the weighted value vectors and get the output of the self-attention layer
2. Matrix calculation of self-attention
    1. Pack embeddings into a matrix `X`, multiply it by the weight matrices `W_Q, W_K, W_V` to get `Q, K, V` matrices: $Q = X @ W_Q, K = X @ W_K, V = X @ W_V$
        - X is N * 512, where N is the number of input words
        - W_Q, W_K, W_V are 512 * 64
        - Q, K, V are N * 64
    2. $Z = softmax(\frac{Q @ K.T}{\sqrt{d_k}})@V$
        - Q @ K.T is N * N
        - Z is N * 64
3. Multi-headed attention
    1. Multiple sets of W_Q/W_K/W_V matrices, each is randomly initialized (Transformer uses 8 sets of Q/K/V weight matrices)
    2. Concat the matrices [Z_0 Z_1 ... Z_7], multiply by an additional matrix `W_O`
        - [Z_0 Z_1 ... Z_7] is N * (64*8)
        - W_O is (64*8) * 512
        - the output matrix is N * 512, has the same size as embediing matrix
4. Positional encoding
    1. `X = t + X`, `t` is the positional encoding of N * 512
    2. Positional encoding: interweave sine and cosine signals?
5. Residuals
    1. Each sub-layer (self-attention, ffnn) in each encoder has a **Residual & Normalize** step
        $Z = LayerNorm(X + Z)$
    2. Normalize keeps every layer's output at mean≈0, variance≈1, so each layer sees well-behaved inputs
    3. Residual add the input back to the output, solve the gradients vanishing problem in deep learning
## Decoder: self-attention + encoder-decoder attention + feed forward
1. Self-attention layer is only allowed to attend to **earlier positions** in the output sequence
2. Encoder-Decoder Attention: creates Q matrix from the layer below it, takes K and V matrix from the output of the encoder stack
    ```
    Q = **decoder_output** @ W_q
    K = encoder_output @ W_k
    V = encoder_output @ W_v
    Z = softmax(Q @ K.T / sqrt(d_k)) @ V
    output = LayerNorm(X + Z)
    ```
3. Final Linear and Softmax layer
    1. The Linear layer is a fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a **logits vector**
    2. For a model with 10_000 vocabs, the logits vector is 10_000 cells wide, each cell corresponding to the score of a unique word
    3. The softmax layer turns scores into **probabilities**, add up to 1.0
    4. The associated word of cell with highest probability is chosen as the output for the time step
## Training loop
1. Loss function
    Cross-entropy: $H(P, Q) = -\Sigma{P_i * log(Q_i)}$
    KL Divergence: $KL(P, Q) = \Sigma{P_i * log(P_i / Q_i)}$
2. Greedy decoding: select the word with the highest probability and throwing away the rest
3. Beam search: select the top `beam_size` words, run the model for `beam_size` times with those words and choose the best 

# Transformer implementation
input token embeddings -- transformer blocks -- output embedding -- softmax -- output probabilities
## Token embedding
1. input: a tensor of token IDs of shape `batch_size * sequence_length`
2. output: a sequence of vectors of shape `batch_size * sequence_length * d_model`
## Pre-norm Transformer block
1. input: a tensor of shape `batch_size * sequence_length * d_model`
2. output: a tensor of shape `batch_size * sequence_length * d_model`
3. layers: pre-norm, self-attention, feed forward
## Output normalization and embedding
1. normalization layer
2. learned linear transformation
    input: a tensor of shape `batch_size * sequence_length * d_model`, output of the last transformer block
    output: predicted next-token logits
## Parameter Initialization
1. Bad initialization can lead to issues like gradients vanishing or exploding
2. Initialization have a siginificant impact on trainding speed and convergence
## Linear Module
1. Implement a Linear class that inherits from `torch.nn.Module` and performs a linear transformation. Your implementation should follow the interface of PyTorch’s built-in `nn.Linear` module, except for not having a `bias` argument or parameter
2. Use `torch.nn.init.trunc_normal_` to initialize the weights

## Embedding Module

# Ref
## Blogs
1. https://jalammar.github.io/illustrated-transformer/ ✅
2. https://nlp.seas.harvard.edu/annotated-transformer/ ✅
3. PyTorch https://docs.pytorch.org/tutorials/index.html
## Paper
1. Attention mechanism https://arxiv.org/abs/1409.0473
2. Attention is all you need https://arxiv.org/abs/1706.03762
## Project
4. minGPT https://github.com/karpathy/minGPT
5. nanoGPT
6. nanoChat https://github.com/karpathy/nanochat
7. https://karpathy.ai/zero-to-hero.html
