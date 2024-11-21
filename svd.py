import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

device = 'cuda:4'
dtype = torch.bfloat16

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/data3/haibo/weights/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=dtype
).to(device)

# Load the conditioning image
image = load_image("rocket in fire.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)