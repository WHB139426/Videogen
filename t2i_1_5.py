from tqdm.auto import tqdm
from PIL import Image  
import torch
import random
import numpy as np
from torch.backends import cudnn
from transformers import CLIPTokenizer  
from diffusers import DDPMScheduler, DDIMScheduler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from models.modeling_clip import CLIPTextModel
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel


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

seed = random.randint(0, 1e9)
init_seeds(seed)
print('seed: ', seed)

# Define parameters
model_path = "/data3/haibo/weights/stable-diffusion-v1-5" 
height = 256 # default height of Stable Diffusion  
width = 256 # default width of Stable Diffusion  
num_inference_steps = 50 # Number of denoising steps  
guidance_scale = 7.5 # Scale for classifier-free guidance  
do_classifier_free_guidance = True
text = "close up photo of a rabbit, forest, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot"
device = "cuda:4"  
dtype = torch.bfloat16
ckpt = "experiments/image_epoch_5_lora.pth"
lora_zero = False

# Load models and scheduler
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)  
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")  
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)  
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)  
if ckpt is not None and 'lora' in ckpt:
    from peft import LoraConfig
    unet_lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
if ckpt is not None:
    unet.load_state_dict(torch.load(ckpt, map_location='cpu'))

if lora_zero:
    for name, param in unet.named_parameters():
        if "lora" in name:
            with torch.no_grad():  # 确保不记录到计算图中
                param.zero_()  # 将参数置为 0

scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

print(get_parameter_number(vae))
print(get_parameter_number(text_encoder))
print(get_parameter_number(unet))

# Encode input prompt
text_inputs = tokenizer([text], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")  
prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0] # [1, 77, 768]
if do_classifier_free_guidance:
    uncond_input = tokenizer([""], padding="max_length", max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt",)
    negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device))[0] # [1, 77, 768]
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device) # [2, 77, 768]
    
# Prepare latent variables  
shape = (1, unet.config.in_channels, int(height) // 8, int(width) // 8,)
latents = torch.randn((1, unet.config.in_channels, height // 8, width // 8),  device=device, dtype=dtype)  
latents = latents * scheduler.init_noise_sigma

# Denoising loop
scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = scheduler.timesteps
for t in tqdm(timesteps):  
    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2).to(device) if do_classifier_free_guidance else latents
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)  
    with torch.inference_mode():  
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False,)[0] 
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

# from models.ddpm import DDPM
# scheduler = DDPM(device=device, n_steps=1000, min_beta=0.00085, max_beta=0.012, beta_schedule='scaled_linear')
# for t in tqdm(range(1000 - 1, -1, -1)):
#     latent_model_input = torch.cat([latents] * 2).to(device) if do_classifier_free_guidance else latents
#     with torch.inference_mode():  
#         noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False,)[0]
#         # perform guidance
#         if do_classifier_free_guidance:
#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#         latents = scheduler.sample_backward_step(latents, t, noise_pred, True)
        
# decode images  
with torch.inference_mode():  
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=None)[0]   
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()  
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()  
    image = image.round().astype("uint8")  
    image = Image.fromarray(image)  
    image.save(f"samples/{text}.png")