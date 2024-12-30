import os

import torch
from diffusers import StableDiffusion3Pipeline

token = os.getenv("HF_TOKEN")
print(token)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16,
    cache_dir="./models",
    use_auth_token=token,
)
pipe = pipe.to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
image.save("capybara.png")
