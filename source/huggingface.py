# setup
# $ mkdir ~/.cache/huggingface/hub/models--gsdf--Counterfeit-V3.0
# $ cd ~/.cache/huggingface/hub/models--gsdf--Counterfeit-V3.0
# $ curl -LO https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fp16.safetensors

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

hub_dir = Path(os.getenv("HOME"))/".cache/huggingface/hub"
model = str("/home/kousei/.cache/huggingface/hub/AsianModel/Brav6.safetensors")
pipe = StableDiffusionPipeline.from_single_file(
    model,
    torch_dtype=torch.float16
).to("cuda")

# EasyNegativeV2 を使いたい場合
pipe.load_textual_inversion(
    "gsdf/Counterfeit-V3.0",
    weight_name="embedding/EasyNegativeV2.safetensors",
    token="EasyNegativeV2"
)

prompt = "woman,70 years old, high resolution, realistic, portrait, japanese, grey hair, smile, traditional, beautiful skin, fat, kimono"
negative_prompt = "EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), split view, grid view, monochrome,(wrinkle:3)"
#negative_prompt="EasyNegativeV2, extra fingers, fewer fingers"

#generator = None
# seed を固定したい場合
generator = torch.Generator()
generator.manual_seed(1234)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
)

result.images[0].save("output.png")
