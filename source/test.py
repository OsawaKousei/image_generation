from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# Stable Diffusion v2のモデル名
model_id = "stabilityai/stable-diffusion-2"

num = 1

for i in range(num):

    # ノイズスケジューラ
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    # 重みのダウンロード & モデルのロード
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
    # GPU使用。（CPUだと生成にかなり時間かかります。というかいつ終わるのか不明。）
    pipe = pipe.to("cuda")

    # 入力テキスト
    prompt = "woman,40 years old,photo,yellow race"
    negative_prompt = "EasyNegative,paintings,sketches,monochrome"
    image = pipe(prompt=prompt,negative_prompt=negative_prompt).images[0]

    image.save(f"output_{i}.png")
