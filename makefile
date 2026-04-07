.PHONY: train eval test tokenize

train-tiny:
	caffeinate -i nohup uv run python cs336_basics/train.py \
		--mode train \
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
		--max_steps 5_000 \
		--log_interval 100 \
		--valid_interval 10 \
		--cp_interval 1000 \
		--device mps \
		--dtype float32 > train_tiny.log 2>&1 &

valid-tiny:
	uv run python cs336_basics/train.py \
		--mode valid \
		--valid_data data/TinyStoriesV2-GPT4-valid-tokens.npy \
		--snapshot_filepath checkpoint/final_model.pt \
		--device mps \
		--dtype float32

decode-tiny:
	uv run python cs336_basics/train.py \
		--mode decode \
		--context_length 256 \
		--snapshot_filepath checkpoint/final_model.pt \
		--vocab_filepath data/TinyStoriesV2-GPT4-train-10_000V-vocab.json \
		--merge_filepath data/TinyStoriesV2-GPT4-train-10_000V-merge.txt \
		--prompt "hello world" \
		--max_tokens 1000 \
		--device mps \
		--dtype float32
