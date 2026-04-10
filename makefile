.PHONY: train-tiny valid-tiny decode-tiny

MODEL ?= "model/final_model_d512_l4_lr_3e-3_b64.pt"
PROMPT ?= Once upon a time,
TEMP ?= 0.1
TOP_P ?= 0.8

visual:
	uv run python cs336_basics/visual.py \
		--snapshot_filepath $(MODEL) \
		--vocab_size 10_000 \
		--d_model 512 \
		--num_heads 16 \
		--num_layers 4 \
		--d_ff 1344 \
		--theta 10_000 \
		--context_length 256 \
		--batch_size 32 \
		--device mps \
		--dtype float32	\
		--post_norm True

# set `--post_norm True` to move RMS normalization of transformer blocks afterward attn and ffn
train-post-norm:
	caffeinate -i nohup uv run python cs336_basics/train.py \
		--mode train \
		--name d512_l4_lr_3e-3_b32_post_norm \
		--train_data data/TinyStoriesV2-GPT4-train-tokens.npy \
		--vocab_filepath data/TinyStoriesV2-GPT4-train-10_000V-vocab.json \
		--merge_filepath data/TinyStoriesV2-GPT4-train-10_000V-merge.txt \
		--checkpoint_dir checkpoint/ \
		--vocab_size 10_000 \
		--d_model 512 \
		--num_heads 16 \
		--num_layers 4 \
		--d_ff 1344 \
		--theta 10_000 \
		--context_length 256 \
		--batch_size 32 \
		--accumulate_batch_size 1 \
		--max_steps 100 \
		--lr_max 3e-3 \
		--lr_min 3e-4 \
		--T_w 500 \
		--T_c 4000 \
		--log_interval 100 \
		--valid_interval 10 \
		--cp_interval 1000 \
		--device mps \
		--post_norm True \
		--dtype float32 \
		> train_tiny_lr3e-3_b32_post_norm.log 2>&1 &

# set `--no_norm True` to remove all RMS normalization layers of transformer blocks
# decrease `lr_max` to 3e-6 to avoid divergence
train-no-norm:
	caffeinate -i nohup uv run python cs336_basics/train.py \
		--mode train \
		--name d512_l4_lr_3e-6_b32_no_norm \
		--train_data data/TinyStoriesV2-GPT4-train-tokens.npy \
		--vocab_filepath data/TinyStoriesV2-GPT4-train-10_000V-vocab.json \
		--merge_filepath data/TinyStoriesV2-GPT4-train-10_000V-merge.txt \
		--checkpoint_dir checkpoint/ \
		--vocab_size 10_000 \
		--d_model 512 \
		--num_heads 16 \
		--num_layers 4 \
		--d_ff 1344 \
		--theta 10_000 \
		--context_length 256 \
		--batch_size 32 \
		--accumulate_batch_size 1 \
		--max_steps 5000 \
		--lr_max 3e-6 \
		--lr_min 3e-7 \
		--T_w 500 \
		--T_c 4000 \
		--log_interval 100 \
		--valid_interval 10 \
		--cp_interval 1000 \
		--device mps \
		--no_norm True \
		--dtype float32 > train_tiny_lr3e-6_b32_no_norm.log 2>&1 &

train-tiny:
	caffeinate -i nohup uv run python cs336_basics/train.py \
		--mode train \
		--name d512_l4_lr_3e-3_b16 \
		--train_data data/TinyStoriesV2-GPT4-train-tokens.npy \
		--vocab_filepath data/TinyStoriesV2-GPT4-train-10_000V-vocab.json \
		--merge_filepath data/TinyStoriesV2-GPT4-train-10_000V-merge.txt \
		--checkpoint_dir checkpoint/ \
		--vocab_size 10_000 \
		--d_model 512 \
		--num_heads 16 \
		--num_layers 4 \
		--d_ff 1344 \
		--theta 10_000 \
		--context_length 256 \
		--batch_size 16 \
		--accumulate_batch_size 1 \
		--max_steps 5000 \
		--lr_max 3e-3 \
		--lr_min 3e-4 \
		--T_w 500 \
		--T_c 4000 \
		--log_interval 100 \
		--valid_interval 10 \
		--cp_interval 1000 \
		--device mps \
		--dtype float32 > train_tiny_lr3e-3_b16.log 2>&1 &

valid-tiny:
	uv run python cs336_basics/train.py \
		--mode valid \
		--valid_data data/TinyStoriesV2-GPT4-valid-tokens.npy \
		--snapshot_filepath $(MODEL) \
		--vocab_size 10_000 \
		--d_model 512 \
		--num_heads 16 \
		--num_layers 4 \
		--d_ff 1344 \
		--theta 10_000 \
		--context_length 256 \
		--batch_size 32 \
		--device mps \
		--dtype float32

decode-tiny:
	uv run python cs336_basics/train.py \
		--mode decode \
		--vocab_size 10_000 \
		--d_model 512 \
		--num_heads 16 \
		--num_layers 4 \
		--d_ff 1344 \
		--theta 10_000 \
		--context_length 256 \
		--batch_size 32 \
		--snapshot_filepath $(MODEL) \
		--vocab_filepath data/TinyStoriesV2-GPT4-train-10_000V-vocab.json \
		--merge_filepath data/TinyStoriesV2-GPT4-train-10_000V-merge.txt \
		--prompt "$(PROMPT)" \
		--max_tokens 1000 \
		--temperature $(TEMP) \
		--top_p $(TOP_P) \
		--device mps \
		--dtype float32
