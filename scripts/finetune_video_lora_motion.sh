CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=21312 train.py \
    --bs 3 \
    --grad_accumulation_steps 6 \
    --epoch 10 \
    --lr 1e-4 \
    --save_interval 0.1 \
    --stage video \
    --num_frames 16 \
    --img_size 256 \
    --n_steps 1000 \
    --min_beta 0.00085 \
    --max_beta 0.012 \
    --cfg_ratio 0.1 \
    --use_lora \