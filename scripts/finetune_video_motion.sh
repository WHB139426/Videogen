CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=21312 train.py \
    --bs 5 \
    --grad_accumulation_steps 4 \
    --epoch 10 \
    --lr 1e-4 \
    --save_interval 0.1 \
    --stage video \
    --num_frames 16 \
    --stride 4 \
    --img_size 256 \
    --n_steps 1000 \
    --min_beta 0.00085 \
    --max_beta 0.012 \
    --beta_schedule linear \
    --cfg_ratio 0.1 \
    --use_lora \
    --lora_rank 32 \
    --lora_path experiments/image_epoch_5_lora_rank_32.pth \