.PHONY: train eval test tokenize

train-tiny:
	uv run python cs336_basics/train.py \
		--mode train
		--train_data data/TinyStoriesV2-GPT4-train-tokens.npy \
		--vocab_filepath data/TinyStoriesV2-GPT4-train-10_000V-vocab.json \
		--merge_filepath data/TinyStoriesV2-GPT4-train-10_000V-merge.txt \
		--checkpoint_dir checkpoint/ \
		--vocab_size 10000 \
		--d_model 512 \
		--num_heads 8 \
		--num_layers 6 \
		--d_ff 2048 \
		--context_length 256 \
		--batch_size 32 \
		--max_steps 100 \
		--log_interval 10 \
		--valid_interval 10 \
		--cp_interval 1000 \
		--device mps
		--dtype float32

valid-tiny:
	uv run python cs336_basics/train.py \
		--mode valid \
		--valid_data data/TinyStoriesV2-GPT4-valid-tokens.npy \
		--snapshot_filepath checkpoint/final_model.pt \
		--device mps \
		--dtype float32

