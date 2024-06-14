torchrun --standalone --nproc_per_node=8 eval_gpt2.py \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --model d12 \
    --checkpoint "/data-4/nano-logs/dbb98246-4c7d-4eb9-8755-71e98278ba5b/*.pt" \
    --batch_size 8 \
    --sequence_length 1024 \
    --val_steps 5
