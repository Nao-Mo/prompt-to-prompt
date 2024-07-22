import torch
import os
import random
from diffusers import DiffusionPipeline, StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
model_id1 = "stabilityai/stable-diffusion-2-1-base"

shape = (1, 4, 64, 64)
prompt = "an image of a car in childrendrwaing style"
directory = "images/sd1.5/"+prompt+"/"
layout = torch.strided
batch_size = 201

pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, requires_safety_checker=False)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=False)
latents = [None]*batch_size
imgs = [None]*batch_size
for i in range(201):
    seed =i
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latents[i] = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16, layout=layout)
    imgs[i] = pipeline(prompt,latents = latents[i], generator = generator).images[0]
    print(type(imgs[i]))
    os.makedirs(directory, exist_ok=True)
    # imgs[i].save("images/origin/photo/A image of a modern car in photo style_seed:"+str(seed)+"_num:"+str(i)+".png")
    imgs[i].save(directory+prompt+"_num:"+str(i)+"_seed:"+str(seed)+".png")