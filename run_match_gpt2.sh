torchrun --standalone --nproc_per_node=8 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M \
    --model d12 \
    --batch_size 64 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --val_loss_every 128 \
    --num_iterations 12288 \
    --weight_decay 0.1 \
    --learning_rate 0.0018 \
    --warmup_iters 256 \
    --warmdown_iters 2048
