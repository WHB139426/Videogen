import torch
import random
import numpy as np
from torch.backends import cudnn
import os
import sys
from tqdm.auto import tqdm
from PIL import Image  
import torch  
from transformers import CLIPTokenizer  
from diffusers import DDPMScheduler, DDIMScheduler
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from models.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel

# from diffusers import StableDiffusionXLPipeline
# pipe = StableDiffusionXLPipeline.from_pretrained("/data3/haibo/weights/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True)
# pipe.to("cuda:0")
# prompt = "A cat holding a sign that says hello world"
# images = pipe(prompt=prompt).images[0]
# images.save(f"{prompt}.png")

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

init_seeds(random.randint(0,1e9))


model_path = "/data3/haibo/weights/stable-diffusion-xl-base-1.0" 
height = 1024 # default height of Stable Diffusion  
width = 1024 # default width of Stable Diffusion  
num_inference_steps = 50 # Number of denoising steps  
guidance_scale = 5 # Scale for classifier-free guidance  
do_classifier_free_guidance = True
text = "rocket in fire"
device = "cuda:0"  
dtype = torch.bfloat16
ckpt = None

# Load models and scheduler
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)    
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")  
tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")  
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)    
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=dtype).to(device)    
tokenizers = [tokenizer, tokenizer_2]
text_encoders = ([text_encoder, text_encoder_2])
unet = UNet2DConditionModel.from_pretrained(  model_path, subfolder="unet", torch_dtype=dtype).to(device)    
if ckpt is not None:
    unet.load_state_dict(torch.load(ckpt, map_location='cpu'))
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

print(get_parameter_number(vae))
print(get_parameter_number(text_encoder))
print(get_parameter_number(text_encoder_2))
print(get_parameter_number(unet))

# Encode input prompt
prompt = [text]
prompt_2 = prompt
prompt_embeds_list = []
prompts = [prompt, prompt_2]
for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds_list.append(prompt_embeds)
prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(dtype=text_encoder_2.dtype, device=device) # [1, 77, 2048]
negative_prompt_embeds = torch.zeros_like(prompt_embeds)
negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds).to(dtype=text_encoder_2.dtype, device=device)
# print(prompt_embeds.shape, negative_prompt_embeds.shape, pooled_prompt_embeds.shape, negative_pooled_prompt_embeds.shape)
# torch.Size([1, 77, 2048]) torch.Size([1, 77, 2048]) torch.Size([1, 1280]) torch.Size([1, 1280])

# Prepare added time ids & embeddings
add_text_embeds = pooled_prompt_embeds
def _get_add_time_ids(
    original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids
add_time_ids = _get_add_time_ids((height, width),(0,0),(height, width),dtype=prompt_embeds.dtype,text_encoder_projection_dim=text_encoder_2.config.projection_dim,) # [[1024., 1024.,    0.,    0., 1024., 1024.]]
negative_add_time_ids = add_time_ids

if do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device) # [2, 77, 2048]
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0).to(device) # [2, 1280]
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0).to(device) # [2, 6]

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
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    with torch.inference_mode():  
        noise_pred = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=prompt_embeds, 
            added_cond_kwargs=added_cond_kwargs, 
            return_dict=False,)[0] 
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

# decode images  
with torch.inference_mode():  
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]   
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()  
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()  
    image = image.round().astype("uint8")  
    image = Image.fromarray(image)  
    image.save(f"{text}.png")