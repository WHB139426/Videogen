import torch
import torch.nn as nn
import contextlib
import random
from einops import rearrange

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models_3d_vae.autoencoder_kl import AutoencoderKL
from mm_utils.utils import *

class AE_1(nn.Module):
    def __init__(self, 
                 dtype=torch.float32,
                 model_path="/home/haibo/weights/stable-diffusion-v1-5",
                 img_size = 512,
                 use_3d = True,
                 frame_num=4,
                 ):
        super().__init__()
        self.dtype = dtype
        self.use_3d = use_3d
        self.image_size = img_size
        self.frame_num = frame_num
        assert frame_num==4

        self.vae = AutoencoderKL(use_3d=use_3d)
        self.vae.load_state_dict(torch.load(os.path.join(model_path, "vae/vae.pth"), map_location='cpu'), strict=False)
        self.vae.to(self.dtype)

        # for name, param in self.vae.named_parameters():
        #     if 'motion_modules' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples):
        """
        * `pixel_values` has shape `[batch_size, frame_num, in_channels, height, width]`, frame_num=4
        """
        pixel_values = samples['pixel_values'] 
        sample = pixel_values[:, :2, :, :, :].permute(0, 2, 1, 3, 4) # [batch_size, in_channels, 2, height, width]
        target = pixel_values[:, 2:4, :, :, :].permute(0, 2, 1, 3, 4) # [batch_size, in_channels, 2, height, width]
        with self.maybe_autocast():
            outputs = self.vae(sample=sample, target=target)
        loss = outputs['loss']
        return loss

# batch_size = 3
# device = 'cuda:0'
# use_3d = True
# frame_num = 4
# img_size = 512

# model = AE_1(dtype=torch.float32 if device=='cpu' else torch.bfloat16, use_3d=use_3d, frame_num=frame_num)
# model.to(device)
# print(get_parameter_number(model))

# from datasets.mix_pretrain import MixPretrain
# from torch.utils.data import DataLoader
# dataset = MixPretrain(
#     img_size=img_size, 
#     num_frames = frame_num, 
#     stride = 4,       
#     anno_path = "/home/haibo/data/mix_pretrain/mix_pretrain.json",
#     video_path = "/home/haibo/data",)
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=16)
# optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = 5e-5)
# scaler = torch.cuda.amp.GradScaler() #训练前实例化一个GradScaler对象
# for step, data in enumerate(data_loader):
#     print(data['pixel_values'].shape)
#     samples = {
#             "pixel_values": data['pixel_values'].to(device),
#             "prompts": data['prompts'],
#         }
#     with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
#         loss = model(samples)
#     print(loss)
#     if model.dtype==torch.float32 or model.dtype==torch.bfloat16:
#         loss.backward()
#         optimizer.step()
#     else:
#         scaler.scale(loss).backward()  #为了梯度放大
#         scaler.step(optimizer)
#         scaler.update()  #准备着，看是否要增大scaler
#     optimizer.zero_grad()         

#     if step==5:
#         break