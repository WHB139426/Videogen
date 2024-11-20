import torch
import torch.nn as nn
import contextlib
import random

from transformers import CLIPTokenizer  

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models.autoencoder_kl import AutoencoderKL
from models.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection
from models.unet_2d_condition import UNet2DConditionModel
from models.ddpm import DDPM
from mm_utils.utils import *
    
class SD_XL(nn.Module):
    def __init__(self, 
                 dtype=torch.float32,
                 model_path="/home/haibo/weights/stable-diffusion-xl-base-1.0",
                 n_steps = 1000,
                 min_beta = 0.00085,
                 max_beta = 0.012,
                 beta_schedule = 'scaled_linear',
                 cfg_ratio = 0.1,
                 img_size = 1024,
                 ):
        super().__init__()
        self.dtype = dtype
        self.image_size = img_size
        self.cfg_ratio = cfg_ratio
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=self.dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")  
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")  
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=self.dtype) 
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=self.dtype)  
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=self.dtype) 
        self.scheduler = DDPM(device=self.device, n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, beta_schedule=beta_schedule)
        self.loss_function = nn.MSELoss()

        for name, param in self.vae.named_parameters():
            param.requires_grad = False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.text_encoder_2.named_parameters():
            param.requires_grad = False
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()

        self.tokenizers = [self.tokenizer, self.tokenizer_2]
        self.text_encoders = [self.text_encoder, self.text_encoder_2]

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

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, batchsize
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids for i in range(batchsize)], dtype=dtype).to(self.device)
        return add_time_ids

    def cfg_discard(self, prompt_embeds, add_text_embeds, add_time_ids):
        for i in range(prompt_embeds.shape[0]):
            if random.random() < self.cfg_ratio:
                prompt_embeds[i] = torch.zeros_like(prompt_embeds[i]).to(self.device)
                add_text_embeds[i] = torch.zeros_like(add_text_embeds[i]).to(self.device)
            else:
                continue
        return prompt_embeds, add_text_embeds, add_time_ids

    def encode_prompt(self, texts):
        prompt = texts
        prompt_2 = prompt
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]

        for prompt, tokenizer, text_encoder in zip(prompts, self.tokenizers, self.text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_embeds = text_encoder(text_inputs.input_ids.to(self.device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(self.device) # [bs, 77, 2048]
        add_text_embeds = pooled_prompt_embeds # [bs, 1280]
        add_time_ids = self._get_add_time_ids((self.image_size, self.image_size),(0,0),(self.image_size, self.image_size),dtype=prompt_embeds.dtype,batchsize=prompt_embeds.shape[0]) # [bs, 6]

        prompt_embeds, add_text_embeds, add_time_ids = self.cfg_discard(prompt_embeds, add_text_embeds, add_time_ids)

        return prompt_embeds, add_text_embeds, add_time_ids

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
                prompt_embeds, add_text_embeds, add_time_ids = self.encode_prompt(samples['prompts'])
        with self.maybe_autocast():
            t = torch.randint(0, self.scheduler.n_steps, (latent.shape[0],)).to(self.device)
            noised_latent, eps = self.scheduler.sample_forward(latent, t)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            eps_theta = self.unet(
                noised_latent, 
                t, 
                encoder_hidden_states=prompt_embeds, 
                added_cond_kwargs=added_cond_kwargs, 
                return_dict=False,)[0] 
            loss = self.loss_function(eps_theta, eps)
        return loss


# batch_size = 4
# device = 'cuda:0'
# model = SD_XL(dtype=torch.bfloat16, img_size=512)
# model.to(device)
# print(get_parameter_number(model))

# from datasets.pokemon import Pokemon
# from torch.utils.data import DataLoader
# dataset = Pokemon()
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=16)
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