## Tier 1: Foundational (read these first)

| Paper | Year | Why it matters |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The Transformer. Everything starts here. |
| [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | Showed scaling up a simple LM works surprisingly well |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | 2018 | The encoder side of the story, defined pre-training + fine-tuning paradigm |

## Tier 2: Scaling & Training (the science of making LLMs work)

| Paper | Year | Why it matters |
|---|---|---|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Kaplan et al. — loss is predictable from model size, data, compute |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | 2022 | Corrected scaling laws — most models were undertrained on data |
| [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | 2020 | Showed in-context learning emerges at scale |
| [LLaMA: Open and Efficient Foundation LMs](https://arxiv.org/abs/2302.13971) | 2023 | Best open recipe — practical training details |
| [LLaMA 2](https://arxiv.org/abs/2307.09288) | 2023 | Training + post-training details, RLHF section is great |

## Tier 3: Architecture Improvements

| Paper | Year | What it introduced |
|---|---|---|
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 2021 | RoPE — what you implemented |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 2019 | RMSNorm — what you implemented |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | IO-aware exact attention — the key systems optimization |
| [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) | 2023 | Grouped-Query Attention — fewer KV heads for efficiency |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 2020 | SwiGLU activation — used in LLaMA and most modern LLMs |

## Tier 4: Alignment & Post-Training

| Paper | Year | Why it matters |
|---|---|---|
| [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) | 2022 | RLHF — how ChatGPT was made |
| [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | 2023 | Simpler alternative to RLHF |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022 | Anthropic's approach to AI safety |

## Tier 5: Efficiency & Systems

| Paper | Year | Why it matters |
|---|---|---|
| [Megatron-LM: Training Multi-Billion Parameter LMs](https://arxiv.org/abs/1909.08053) | 2019 | How to parallelize across GPUs |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 2019 | Distributed optimizer — DeepSpeed |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | Efficient fine-tuning |

## Tier 6: Understanding & Emergent Abilities

| Paper | Year | Why it matters |
|---|---|---|
| [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) | 2023 | Comprehensive overview of everything |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 2022 | "Let's think step by step" works |
| [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) | 2022 | Abilities that appear only at scale |

## Suggested reading order

```
Week 1:  Attention Is All You Need
Week 2:  GPT-2, GPT-3
Week 3:  Scaling Laws, Chinchilla
Week 4:  LLaMA, LLaMA 2
Week 5:  FlashAttention, RoPE
Week 6:  InstructGPT, DPO
Week 7:  The survey paper (fills in gaps)
```

The **top 5 must-reads** if you have limited time:

1. **Attention Is All You Need** — the architecture
2. **Chinchilla** — how to train efficiently
3. **LLaMA** — practical modern recipe
4. **FlashAttention** — the systems breakthrough
5. **InstructGPT** — how base models become ChatGPT