import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from torch.backends import cudnn
from mm_utils.utils import *
from mm_utils.optims import *


# nohup bash scripts/finetune_image_lora.sh > finetune_image_lora.out 2>&1 &
# nohup bash scripts/finetune_video_lora_motion.sh > finetune_video_lora_motion.out 2>&1 &

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dtype', default=torch.bfloat16)
    parser.add_argument('--bs', type=int, default=3)
    parser.add_argument('--grad_accumulation_steps', type=int, default=5) # overall: world_size*bs*grad_accumulation_steps = 64
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5) 
    parser.add_argument('--save_interval', type=float, default=1/10, help='save per save_interval epoch')
    parser.add_argument('--use_lora', action='store_true')

    parser.add_argument('--stage', type=str, default='video', choices=['image', 'video'])
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--stride', type=int, default=-1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--min_beta', type=float, default=0.00085)
    parser.add_argument('--max_beta', type=float, default=0.012)
    parser.add_argument('--beta_schedule', type=str, default='scaled_linear', choices=['scaled_linear', 'linear'])
    parser.add_argument('--cfg_ratio', type=float, default=0.1)

    parser.add_argument('--use_schedule', type=bool, default=False)
    parser.add_argument('--warm_up_epoches', type=int, default=0.03, help='epoches from warmup_start_lr->lr')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6)
    parser.add_argument('--min_lr', type=float, default=1e-7, help='min_lr for consine annealing')
    parser.add_argument('--max_T', type=int, default=1e9, help='epoches for lr->min_lr / min_lr->lr')

    args = parser.parse_args()
    return args

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def reduce_metric(metric):
    metric_tensor = torch.tensor(metric).cuda(dist.get_rank())
    dist.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
    metric = metric_tensor.item() / dist.get_world_size()
    return metric

def plot_records(record_list, record_type):
    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(record_list,'b',label=record_type)
    plt.ylabel(record_type)
    plt.xlabel('iteration')
    plt.legend()
    plt.savefig(f"{record_type}.jpg")
    plt.close('all')

def train(args, model, train_dataset, rank):

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler, pin_memory=True, shuffle=False, drop_last=True, num_workers=4)

    optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = args.lr)
    lr_schedule = LinearWarmupCosineLRScheduler(optimizer, max_epoch=args.max_T, min_lr=args.min_lr, init_lr=args.lr, warmup_steps=int(len(train_loader)*args.warm_up_epoches), warmup_start_lr=args.warmup_start_lr)

    scaler = torch.cuda.amp.GradScaler() #训练前实例化一个GradScaler对象
    iteration_loss_list = []
    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

    for epoch in range(args.epoch):

        model.train()
        # 设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
        train_loader.sampler.set_epoch(epoch)
        iteration_loss = 0

        for train_idx, data in enumerate(tqdm(train_loader)):
            with torch.cuda.amp.autocast(enabled=True, dtype=model.module.dtype): # 前后开启autocast
                samples = {
                    'pixel_values': data['pixel_values'].cuda(rank),
                    'prompts': data['prompts']
                }
                loss = model(samples)

            normalized_loss = loss / args.grad_accumulation_steps
            if args.dtype==torch.float32 or args.dtype==torch.bfloat16:
                normalized_loss.backward()
            else:
                scaler.scale(normalized_loss).backward()  #为了梯度放大
            iteration_loss += normalized_loss.item() 

            if (train_idx + 1) % args.grad_accumulation_steps == 0:
                if args.dtype==torch.float32 or args.dtype==torch.bfloat16:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()

                if args.use_schedule:
                    lr_schedule.step(cur_epoch=epoch, cur_step=train_idx)

                optimizer.zero_grad()    

                iteration_loss_list.append(reduce_metric(iteration_loss))
                iteration_loss = 0

            plot_interval = int(steps_per_epoch*args.grad_accumulation_steps/100)
            if (train_idx + 1) % plot_interval == 0:
                if rank == 0:
                    plot_records(iteration_loss_list, f'{args.stage}_iteration_loss')

            if (train_idx+1) % int(args.save_interval*steps_per_epoch*args.grad_accumulation_steps) == 0:
                if rank == 0 and args.stage == 'video':
                    trainable_state_dict = {
                            k: v for k, v in model.module.unet.state_dict().items()
                            if v.requires_grad
                        }
                    print("save an interval ckpt!")
                    torch.save(trainable_state_dict, f'./experiments/{args.stage}_epoch_{epoch+1}_iteration_{train_idx+1}.pth')

        if rank == 0:
            trainable_state_dict = {
                k: v for k, v in model.module.unet.state_dict().items()
                if v.requires_grad
            }
            print('epoch: ', epoch+1, ' train_loss: ', sum(iteration_loss_list)/len(iteration_loss_list))
            torch.save(trainable_state_dict, f'./experiments/{args.stage}_epoch_{epoch+1}.pth')


def main_worker(args):

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    init_seeds(args.seed + rank)

    from datasets.mix_pretrain import MixPretrain
    train_dataset = MixPretrain(
        img_size=args.img_size, 
        num_frames = 1 if args.stage == 'image' else args.num_frames, 
        stride = args.stride,       
        anno_path = "/data3/haibo/data/mix_pretrain/mix_pretrain.json",
        video_path = "/data3/haibo/data",)

    from models.stable_diffusion_1_5 import SD_1_5
        if args.stage == 'image':
            use_3d = False
        elif args.stage == 'video':
            use_3d = True
        model = SD_1_5(
            dtype=args.dtype, model_path="/data3/haibo/weights/stable-diffusion-v1-5", img_size=args.img_size, use_lora=args.use_lora, use_3d=use_3d,
            n_steps=args.n_steps, min_beta=args.min_beta, max_beta=args.max_beta, cfg_ratio=args.cfg_ratio, beta_schedule = args.beta_schedule,)
        # if use_3d:
        #     model.unet.load_state_dict(torch.load("experiments/video_epoch_3_iteration_8032_lora_no_stride.pth", map_location='cpu'), strict=False)

    model = torch.nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank])

    if rank == 0:
        print(get_parameter_number(model), 'train samples: ', len(train_dataset))
        print(args)

    train(args, model, train_dataset, rank)

    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main_worker(args)