import torch
import torch.nn as nn
import contextlib
import random

from transformers import CLIPTokenizer  

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models.autoencoder_kl import AutoencoderKL
from models.modeling_clip import CLIPTextModel
from models.unet_2d_condition import UNet2DConditionModel
from models.ddpm import DDPM
from mm_utils.utils import *
    
class SD_1_5(nn.Module):
    def __init__(self, 
                 dtype=torch.float32,
                 model_path="/home/haibo/weights/stable-diffusion-v1-5",
                 n_steps = 1000,
                 min_beta = 0.00085,
                 max_beta = 0.012,
                 beta_schedule = 'scaled_linear',
                 cfg_ratio = 0.1,
                 img_size = 512,
                 use_lora = False,
                 ):
        super().__init__()
        self.dtype = dtype
        self.image_size = img_size
        self.cfg_ratio = cfg_ratio
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=self.dtype)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")  
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=self.dtype) 
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=self.dtype) 
        self.scheduler = DDPM(device=self.device, n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, beta_schedule=beta_schedule)
        self.loss_function = nn.MSELoss()

        for name, param in self.vae.named_parameters():
            param.requires_grad = False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        self.vae.eval()
        self.text_encoder.eval()

        if use_lora:
            from peft import LoraConfig
            unet_lora_config = LoraConfig(
                r=128,
                lora_alpha=256,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)

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

    def cfg_discard(self, text):
        return ['' if random.random() < self.cfg_ratio else item for item in text]

    def encode_prompt(self, texts):
        texts = self.cfg_discard(texts)
        text_inputs = self.clip_tokenizer(
            texts, 
            padding="max_length", 
            max_length=self.clip_tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
            )  
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.device))[0] # [bs, 77, 768]
        return prompt_embeds

    def forward(self, samples):
        """
        * `prompts` [batch_size]
        * `pixel_values` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        pixel_values = samples['pixel_values']
        with self.maybe_autocast():
            with torch.no_grad():
                latent = self.vae.encode(pixel_values).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
                prompt_embeds = self.encode_prompt(samples['prompts'])

        with self.maybe_autocast():
            t = torch.randint(0, self.scheduler.n_steps, (latent.shape[0],)).to(self.device)
            noised_latent, eps = self.scheduler.sample_forward(latent, t)
            eps_theta = self.unet(noised_latent, t, encoder_hidden_states=prompt_embeds, return_dict=False,)[0]
            loss = self.loss_function(eps_theta, eps)
        return loss


# batch_size = 4
# device = 'cpu'
# model = SD_1_5(torch.float32, use_lora=True)
# model.to(device)
# print(get_parameter_number(model))

# from datasets.pokemon import Pokemon
# from torch.utils.data import DataLoader
# dataset = Pokemon()
# data_loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=16)
# optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = 5e-5)
# scaler = torch.cuda.amp.GradScaler() #训练前实例化一个GradScaler对象
# for step, data in enumerate(data_loader):
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