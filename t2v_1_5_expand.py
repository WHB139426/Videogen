from tqdm.auto import tqdm
from PIL import Image  
import torch
import random
import numpy as np
from torch.backends import cudnn
from transformers import CLIPTokenizer  
from diffusers import DDPMScheduler, DDIMScheduler
import sys
from moviepy.editor import ImageSequenceClip
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *

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
init_seeds(42)
print('seed: ', seed)

texts = [
    "A train going down the train track",
]

images = [
    'samples/A train going down the train track.png',
]

# Define parameters
model_path = "/home/haibo/weights/stable-diffusion-v1-5/" 
height = 256 # default height of Stable Diffusion  
width = 256 # default width of Stable Diffusion 
num_frames = 16 
fps = 4
num_inference_steps = 50 # Number of denoising steps  
guidance_scale = 7 # Scale for classifier-free guidance  
do_classifier_free_guidance = True
device = "cuda:6"  
dtype = torch.float32 if device=='cpu' else torch.bfloat16
ckpt = None
stage = 'expand'
expand_conv_in = True

# Load models and scheduler
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
from models_expand.stable_diffusion_1_5 import SD_1_5
model = SD_1_5(dtype=dtype, model_path=model_path, img_size=height, expand_conv_in=expand_conv_in)
if ckpt != None:
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
model.to(device)

vae = model.vae 
tokenizer = model.clip_tokenizer
text_encoder = model.text_encoder
unet = model.unet
print(get_parameter_number(vae))
print(get_parameter_number(text_encoder))
print(get_parameter_number(unet))

vae_processor = image_transform(image_size=height, mean=0.5, std=0.5)

for step, (image, text) in enumerate(zip(images, texts)):
    # Encode input prompt
    text_inputs = tokenizer([text], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")  
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0] # [1, 77, 768]

    if do_classifier_free_guidance:
        uncond_input = tokenizer([""], padding="max_length", max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt",)
        negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device))[0] # [1, 77, 768]
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device) # [2, 77, 768]

    encoder_hidden_states = prompt_embeds

    frames = []

    for index in range(num_frames):

        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            vae_pixel_values = vae_processor(load_image(image)).unsqueeze(0).to(device).to(dtype) # [1, 3, 512, 512]

        # Prepare latent variables  
        shape = (1, unet.config.in_channels, int(height) // 8, int(width) // 8,)
        latents = torch.randn((1, unet.config.in_channels, height // 8, width // 8),  device=device, dtype=dtype)  
        latents = latents * scheduler.init_noise_sigma # [1, 4, 64, 64]
        with torch.no_grad():
            cond_latent = vae.encode(vae_pixel_values).latent_dist.sample() * vae.config.scaling_factor # [1, 4, 64, 64]

        # Denoising loop
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        for t in tqdm(timesteps):  
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2).to(device) if do_classifier_free_guidance else latents # [2, 4, 64, 64]
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)  
            with torch.inference_mode():  
                # predict the noise residual
                with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                    latent_model_input = torch.cat([latent_model_input, cond_latent.repeat(latent_model_input.shape[0], 1, 1, 1)], dim=1) # [2, 8, 64, 64] 
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states, return_dict=False,)[0] # [2, 4, 64, 64]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # [1, 4, 64, 64]
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
        # decode images  
        with torch.inference_mode():  
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=None)[0]   
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()  
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()  
        image = image.round().astype("uint8")  
        frames.append(image)
        image = Image.fromarray(image)  
        image.save(f"samples/temp/{index}.png")
        image = f"samples/temp/{index}.png"
    
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(f"samples/{text}.mp4", codec='libx264')
