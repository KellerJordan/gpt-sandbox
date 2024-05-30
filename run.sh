torchrun --standalone --nproc_per_node=8 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_valid_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M \
    --model d12 \
    --batch_size 64 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --val_loss_every 943 \
    --num_iterations 18865 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0006 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --overfit_single_batch 0
