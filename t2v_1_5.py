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
from einops import rearrange
from moviepy.editor import ImageSequenceClip
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from models.modeling_clip import CLIPTextModel
from models.autoencoder_kl import AutoencoderKL
from models.unet_3d_condition import UNet3DConditionModel, unet_additional_kwargs


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
    "A horse galloping through van Gogh's 'Starry Night'",
    "car running on the road, professional shot, 4k, highly detailed",
    "close up photo of a rabbit, forest, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot",
    "photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
    "Robot dancing in times square",
    "b&w photo of 42 y.o man in black clothes, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
    "Snow rocky mountains peaks canyon. Snow blanketed rocky mountains surround and shadow deep canyons",
    "A drone view of celebration with Christma tree and fireworks, starry sky - background",
    "Pacific coast, carmel by the sea ocean and waves",
]

# Define parameters
model_path = "/data3/haibo/weights/stable-diffusion-v1-5" 
height = 256 # default height of Stable Diffusion  
width = 256 # default width of Stable Diffusion  
frame_num = 16
fps = 4
num_inference_steps = 50 # Number of denoising steps  
guidance_scale = 7.5 # Scale for classifier-free guidance  
do_classifier_free_guidance = True

device = "cuda:4"  
dtype = torch.float32 if device else torch.bfloat16
ckpt = 'experiments/video_epoch_1_iteration_8032_lora.pth'
lora_alpha = 1 # [0, 1] to control lora effect

# Load models and scheduler
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", beta_schedule='scaled_linear')
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)  
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")  
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)  
unet = UNet3DConditionModel(sample_size=width // 8, cross_attention_dim=768, **unet_additional_kwargs) 
if ckpt is not None and 'lora' in ckpt:
    from peft import LoraConfig
    target_modules = []
    for name, param in unet.named_parameters():
        if any(keyword in name for keyword in ['to_q', 'to_k', 'to_v', 'to_out.0']) and 'motion_modules' not in name:
            target_modules.append(name.replace('.weight','').replace('.bias',''))
    target_modules = list(set(target_modules))
    unet_lora_config = LoraConfig(
        r=128,
        lora_alpha=256*lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    unet.add_adapter(unet_lora_config)
if ckpt is not None:
    unet.load_state_dict(torch.load(ckpt, map_location='cpu'))
unet = unet.to(device) 

print(get_parameter_number(vae))
print(get_parameter_number(text_encoder))
print(get_parameter_number(unet))


for text in texts:
    # Encode input prompt
    text_inputs = tokenizer([text], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")  
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0] # [1, 77, 768]
    if do_classifier_free_guidance:
        uncond_input = tokenizer([""], padding="max_length", max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt",)
        negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device))[0] # [1, 77, 768]
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device) # [2, 77, 768]
        prompt_embeds.to(dtype)
        
    # Prepare latent variables  
    shape = (frame_num, unet.config.in_channels, int(height) // 8, int(width) // 8,)
    latents = torch.randn(shape,  device=device, dtype=dtype)  
    latents = latents * scheduler.init_noise_sigma

    # Denoising loop
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    for t in tqdm(timesteps):  
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2).to(device) if do_classifier_free_guidance else latents # [2*frame_num, 4, 32, 32]
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)  
        with torch.inference_mode():  
            # predict the noise residual
            latent_model_input = rearrange(latent_model_input, "(b f) c h w -> b c f h w", f=frame_num) # [2, 4, frame_num, 32, 32]
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False,)[0] # [2, 4, frame_num, 32, 32]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # [1, 4, frame_num, 32, 32] 
            # compute the previous noisy sample x_t -> x_t-1
            noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0] # [frame_num, 4, 32, 32]

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
        image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()  
        image = image.round().astype("uint8")  # [frame_num, 256, 256, 3]


    frames = [frame for frame in image] 
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(f"samples/{text}.mp4", codec='libx264')