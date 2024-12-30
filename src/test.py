import torch
from diffusers import (
    BitsAndBytesConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)

model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    cache_dir="./models",
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, transformer=model_nf4, torch_dtype=torch.bfloat16, cache_dir="./models"
)
pipeline.enable_model_cpu_offload()

prompt = "A capybara holding a sign that reads Hello World"

image = pipeline(
    prompt=prompt,
    num_inference_steps=40,
    guidance_scale=4.5,
    max_sequence_length=512,
).images[0]
image.save("./result/output.png")
