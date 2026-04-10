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
input token embeddings -- transformer blocks -- output embedding(norm + linear) -- softmax -- output probabilities
## Parameter Initialization
1. Bad initialization can lead to issues like gradients vanishing or exploding
2. Initialization have a siginificant impact on trainding speed and convergence
3. Use `torch.nn.init.trunc_normal_` to initialize the weights
## Embedding Module
1. Map each token ID to its associated vector
    1. input: a tensor of token IDs of shape `batch_size * sequence_length` 2. output: a sequence of vectors of shape `batch_size * sequence_length * d_model`
2. Implemented in `embedding.py`
## Pre-norm Transformer block
1. Pre-norm vs post-norm: 
    1. "Attention is all you need" used post-norm
    2. Researchers found pre-norm (with an additional layer normalization after the final Transformer block) **improves Transformer training stability**
2. LayerNorm vs RMSNorm 
    1. $LayerNorm(x) = \gamma * \frac{x - mean}{\sqrt{var + \epsilon}} + \beta$, learned parameter is $\gamma$, $\beta$
    2. $RMSNorm(x) = \gamma * \frac{x}{\sqrt{mean(x^2)} + \epsilon}$
    3. Implemented in `rms_normalization.py`
3. Feed-Forward Network
    1. $ReLU(x) = max(0, x)$
    2. $SiLU(x) = x*\gamma(x) = \frac{x}{1 + e^{-x}}$, similar to ReLU but smooth at zero point
    3. $GLU(x, W_1, W_2) = \gamma(W_1x) * (W_2x)$, where x represents elementwise multiplication here
    4. $FFN(x) = SwiGLU(x, W_1, W_2, W_3) = W_2(SiLU(W_1x)*W_3x)$
    5. Experiments shows that **SwiGLU outperforms baselines** like ReLU and SiLU (without gating) on language modeling tasks
    6. Implemented in `swiglu_feed_forward.py`
4. Positional embeddings
    1. Rotary position embeddings, RoPE
    2. For a given query token $q^i$ at position i, apply a pairwise rotation matrix $R^i$, rotating pairs of embedding elements $q_{2k-1:2k}^{(i)}$ as 2d vectors by angle $\theta_{i,k}=\frac{i}{\theta^{(2k-2)/d}}$ for $k \in {1,...,d/2}$
    3. Implemented in `rope.py`
5. Multihead self attention 
    1. Scaled dot-product attention, implemented in `attention.py`
    2. Multi-head self attention
        1. reshaping Q, K, V from (batch, seq_len, h * d_k) to (batch, h, seq_len, d_k), batching matrix multiplication for multi-head
        2. implemented in `attention.py`
## Output normalization and embedding
1. normalization layer: `rms_normalization.py`
2. output embedding: implemented in `linear.py`
    input: a tensor of shape `(batch_size, seq_len, d_model)`, output of the last transformer block
    output: predicted next-token logits of shape `(batch_size, seq_len, vocab_size)`
## Answers: Transformer LM resource accounting
1. Consider GPT-2 XL, which has the following configuration:
    vocab_size : 50,257
    context_length : 1,024
    num_layers : 48
    d_model : 1,600
    num_heads: 25
    d_ff: 6,400
    Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
    1. Number of trainable parameters 
        Token embedddings: vocab_size * d_model
        RMS normalization: d_model
        Multi-head self attention: 4 * d_model ^ 2
        Feed forward: 3 * d_model * d_ff
        Ouput embeddings: d_model * vocab_size
        Total: $vocab_size\times d + layers(4d^2 + 3dd_{ff} + 2d) + d + d\times{vocab\_size}$, 1,635,844,802 parameters
    2. Required memory to load the model: 6GB for FP32

2. Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.
    - $FLOPs = layers * (4n^2d + 6nd^2 + 6nd{d_{ff}}) + 2nd(vocab\_size)$
    - n = context_length = 1024
    - d_model = 1600
    - d_ff = 6400
    - vocab_size = 50,257
    - layers = 48
    - FLOPs = 4,261,678,284,800
3. Based on your analysis above, which parts of the model require the most FLOPs?
    - ffn: 3,019,898,880,000  FLOPs
    - attn: 1,077,097,267,200 FLOPs
    - output embeddings: 164,682,137,600 FLOPs 
4. Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
    - GPT-2 small 
    - GPT-2 medium
    - GPT-2 large
5. Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change
    - output embedding costs 2,634,914,201,600 FLOPs
    - ffn costs 48,318,382,080,000 FLOPs
    - attn costs 94,542,967,603,200 FLOPs
    - total costs 145,496,263,884,800 FLOPs
    - attention, since $4n^2d$ grows quadratically
# Transformer Training
- Loss function: cross entropy
- Optimizer: AdamW
- Traning loop: infrastructure to load data, save checkpoints and manage traning
## Loss function
1. Cross-entroy loss
2. Implemented in `loss_function.py`
## Optimizer
1. SGD optimizer
    $\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla(\theta_t;B_t)$
    $B_t$ is a random batch of data sampled from dataset D
    $\alpha_t$ is learning rate
2. AdamW optimizer
    gradient: $g \leftarrow \nabla_{\theta}l(\theta;B_t)$
    first moment estimate: $m \leftarrow \beta_1m + (1 - \beta_1)g$
    second mement estimate: $v \leftarrow \beta_2v + (1 - \beta_2)g^2$
    $\alpha_t \leftarrow \alpha \frac{\sqrt{1-(\beta_2)^t}}{1-(\beta_1)^t}$
    $\theta \leftarrow \theta - \alpha_t\frac{m}{\sqrt{v+\epsilon}}$
    $\theta \leftarrow \theta - \alpha\lambda\theta$
3. Implemented in `adamw_optimizer.py`
4. Problem: Resource accounting for training with AdamW
    1. How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the `batch_size` and the model hyperparameters `(vocab_size, context_length, num_layers, d_model, num_heads)`. Assume d_ff = 4 × d_model.
        - Parameters 
            Token embedddings: $V\times d$
            RMS normalization: $d$
            Multi-head self attention: $4d^2$
            Feed forward: $12d^2$
            Ouput embeddings: $V \times d$
            Parameter count: $2Vd + 16Ld^2$
        - Activations
        - Gradients: same as parameter, $2Vd + 16Ld^2$
        - Optimizer state: first and second memoent for each parameter, $4Vd + 32Ld^2$ 
    2. Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory
    3. How many FLOPs does running one step of AdamW take?
    4. Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput. An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single A100

# Training loop
1. Data loader
    Implemented in `utils.py::data_loading`
    Use `mmap` for large dataset
    Specify `dtype` matched the input data
2. Checkpoint
    Iteration number
    Model weights
    Optimizer states
    Save: `torch.save(obj, dest)` dumps an object to a file
    Load: `torch.load(src)`
# Experiment
1. Train on TinyStoriesV2-GPT4 dataset with 5000 steps
    log: `Train-tiny.log`
    final model: `final_model_22M.pt`
    loss: 1.8
    Hyperparameters: 
    ```
    --vocab_size 10_000 \
    --d_model 512 \
    --num_heads 16 \
    --num_layers 4 \
    --d_ff 1344 \
    --theta 10_000 \
    --context_length 256 \
    --batch_size 32 \
    --max_steps 5_000 \
    --log_interval 100 \
    --valid_interval 10 \
    --cp_interval 1000 \
    ```
2. Training record https://wandb.ai/yangzf23-independent-developer/cs336-assignment1/runs/4el28xec?nw=nwuseryangzf23
3. Hyperparameters 
    1. Learning rate variation: include at least one divergent run
        `lr = 3e-4`
        `lr = 3e-3`: 1.8+loss
        `lr_max = 3e-3 with scheduler`: 1.6+loss
        `lr_max = 1e-2 with scheduler`: 1.6+loss
        `lr_max = 1e-1 with scheduler`: divergence
        `lr_max = 3e-5 with scheduler`: 3.1+loss
    2. Batch size variation`
        `batch_size = 32`: 1.6+loss, baseline
        `batch_size = 64`: 0.8+loss
        `batch_size = 16`: 1.8+loss
    3. Decode
        `temperature`: 0.1-1 fluent, 10 gibberish, 0.1 is fine
        `top_p`: 1 gibberish, 0.1 deterministic, 0.8 is fine
3. Architecture modification
    Remove RMSNorm
        Run with `lr=3e-3`, diverge at step 21
        Run with `lr=3e-5`, diverge at step 169
        Run with `lr=3e-7`, loss fluctuates around 9.2 and not falls in 1k steps
        Run with `lr=3e-6`, still **diverge** at step 520
    Replace with post-norm
        Run with `d512, l4, lr3e-3, b32`, post-norm
        Learning curve is actually **similar** to pre-norm, reaches a 1.7+loss. 
        Possible reason: for a shallow model with `l=4`, the post-norm effect could be ignored 
    NoPE vs RoPE
        Run with `d512, l4, lr3e-3, b32`
        Learning curve **converges slower** than RoPE
        ![alt text](<wandb_nope_vs_rope.png>)
    SwiGLU vs SiLU
        Run with `d512, l4, lr3e-3, b32`

4. Training on OpenWebText dataset
5. Modify architecture/hyperparameters for better performance




# Ref
## Blogs
1. https://jalammar.github.io/illustrated-transformer/ ✅
2. https://nlp.seas.harvard.edu/annotated-transformer/ ✅
3. PyTorch https://docs.pytorch.org/tutorials/index.html
## Paper
1. Attention mechanism https://arxiv.org/abs/1409.0473
2. Attention is all you need https://arxiv.org/abs/1706.03762
3. see `paper.md`
## Project
4. minGPT https://github.com/karpathy/minGPT
5. nanoGPT
6. nanoChat https://github.com/karpathy/nanochat
7. https://karpathy.ai/zero-to-hero.html
8. Micrograd by Karpathy

## Books
1. DL
    Dive into Deep Learning https://d2l.ai/
    Deep Learning (Goodfellow)
2. RL: Reinforcement Learning: An Introduction