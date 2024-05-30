# Yields 3.2856 perplexity in 6.44B tokens, which is within +/- 0.001 of what the original
# llm.c trainer gets in 10B tokens, and is roughly GPT-2 (124M) level quality.
# The efficiency gain over the original trainer is due to (1) an increased learning rate,
# (2) half the batch size, and (3) an improved learning rate schedule.
# Assuming this is now properly tuned, it can serve as a baseline for comparing other optimizers
# to AdamW.
torchrun --standalone --nproc_per_node=8 base_train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --total_batch_size 262144 \
    --val_loss_every 128 \
    --num_iterations 24576 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0015 \
    --warmup_iters 256 \
    --overfit_single_batch 0

