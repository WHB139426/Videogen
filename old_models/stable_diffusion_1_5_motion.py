import torch
import torch.nn as nn
import contextlib
import random
from einops import rearrange

from transformers import CLIPTokenizer  

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models.autoencoder_kl import AutoencoderKL
from models.modeling_clip import CLIPTextModel
from models.unet_3d_condition import UNet3DConditionModel, unet_additional_kwargs
from models.ddpm import DDPM
from mm_utils.utils import *



# from models.unet_2d_condition import UNet2DConditionModel
# unet = UNet2DConditionModel.from_pretrained("/home/haibo/weights/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float32)
# # torch.save(unet.state_dict(), '/home/haibo/weights/stable-diffusion-v1-5/unet/unet.pth')
# ckpt = torch.load('/home/haibo/weights/stable-diffusion-v1-5/unet/unet.pth', map_location='cpu')
# unet_3d = UNet3DConditionModel(sample_size=64, cross_attention_dim=768, **unet_additional_kwargs)
# load_status = unet_3d.load_state_dict(torch.load('/home/haibo/weights/stable-diffusion-v1-5/unet/unet.pth', map_location='cpu'), strict=False)
# missing_keys = load_status.missing_keys
# unexpected_keys = load_status.unexpected_keys
# print("Missing keys (not loaded into unet_3d):")
# for missing_key in missing_keys:
#     if 'motion' not in missing_key:
#         print(missing_key)
# print("\nUnexpected keys (present in ckpt but not in unet_3d):")
# print(unexpected_keys)


class SD_1_5_Video(nn.Module):
    def __init__(self, 
                 dtype=torch.float32,
                 model_path="/home/haibo/weights/stable-diffusion-v1-5",
                 n_steps = 1000,
                 min_beta = 0.00085,
                 max_beta = 0.012,
                 beta_schedule = 'linear',
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

        self.unet = UNet3DConditionModel(sample_size=self.image_size // 8, cross_attention_dim=768, **unet_additional_kwargs)
        self.unet.load_state_dict(torch.load(os.path.join(model_path, 'unet/unet.pth'), map_location='cpu'), strict=False)
        self.unet.to(self.device)

        self.scheduler = DDPM(device=self.device, n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, beta_schedule=beta_schedule)
        self.loss_function = nn.MSELoss()

        for name, param in self.vae.named_parameters():
            param.requires_grad = False
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        self.vae.eval()
        self.text_encoder.eval()

        target_modules = []
        for name, param in self.unet.named_parameters():
            if any(keyword in name for keyword in ['to_q', 'to_k', 'to_v', 'to_out.0']) and 'motion_modules' not in name:
                target_modules.append(name.replace('.weight','').replace('.bias',''))
        target_modules = list(set(target_modules))

        if use_lora:
            from peft import LoraConfig
            unet_lora_config = LoraConfig(
                r=128,
                lora_alpha=256,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            self.unet.add_adapter(unet_lora_config)

        for name, param in self.unet.named_parameters():
            if 'motion_modules' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

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
        * `pixel_values` has shape `[batch_size, frame_num, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        pixel_values = samples['pixel_values']
        batch_size, frame_num, _, _, _ = pixel_values.shape
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        with self.maybe_autocast():
            with torch.no_grad():
                latent = self.vae.encode(pixel_values).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
                prompt_embeds = self.encode_prompt(samples['prompts'])

        # latent = rearrange(latent, "(b f) c h w -> b c f h w", f=frame_num)

        with self.maybe_autocast():
            t = torch.randint(0, self.scheduler.n_steps, (batch_size,)).to(self.device) # [batch_size]
            repeat_t = t.unsqueeze(1).repeat(1, frame_num)
            repeat_t = rearrange(repeat_t, "b f -> (b f)")
            noised_latent, eps = self.scheduler.sample_forward(latent, repeat_t)
            noised_latent = rearrange(noised_latent, "(b f) c h w -> b c f h w", f=frame_num)
            eps = rearrange(eps, "(b f) c h w -> b c f h w", f=frame_num)
            eps_theta = self.unet(noised_latent, t, encoder_hidden_states=prompt_embeds, return_dict=False,)[0]
            loss = self.loss_function(eps_theta, eps)
        return loss


# batch_size = 2
# num_frames = 16
# img_size = (320, 576)
# device = 'cpu'
# model = SD_1_5_Video(torch.float32, use_lora=False)
# model.to(device)
# print(get_parameter_number(model))

# from datasets.mix_pretrain import MixPretrain
# from torch.utils.data import DataLoader
# dataset = MixPretrain(num_frames=num_frames, img_size=img_size)
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