import torch
from PIL import Image
import numpy as np
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline

def invert_mask(mask: Image.Image):
    return Image.fromarray(255 - np.array(mask))

# Modify following paths
img_path = "/home/enesmsahin/enes_workspace/data/test/resized_processed/2048x2048_Single_4W-02_smaller.png" # 4 Channel RGBA image
guidance_img_path = "/home/enesmsahin/enes_workspace/data/test/guiding_images/target_guiding_rgb2.png" # 3 channel Guidance image
out_path = "/home/enesmsahin/enes_workspace/guided_inpainting/outputs/guided/test/test/target7" # Output folder

Path.mkdir(Path(out_path), exist_ok=True, parents=True)

device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16").to(device)

seed = 45576

# Modify these parameters
params = {
    "num_inference_steps": 50,
    "guidance_scale": 15,
    "num_images_per_prompt": 4,
    "strength": 0.97,
    "generator": torch.Generator(device=device).manual_seed(seed)
}

prompts = [
    "A beautiful living room with a painting on the wall",
]

defaults = "hyper realism, detailed, realistic, 4k" # Add these default keywords to all prompts
prompts = [p + ", " + defaults for p in prompts]

# negative_prompts = None
negative_prompts = ["cartoon, pastel colors"] * len(prompts)

# Read and resize images
segmented_img = Image.open(img_path).resize((512,512))
guidance_img = Image.open(guidance_img_path).resize((512,512)) # If guidance image is set to None, standard inpainting (no guidance) is applied.
# guidance_img = None

prefix = "un" if guidance_img is None else "" # output file name prefix

input_img = segmented_img.convert("RGB")
original_mask = segmented_img.split()[-1] # alpha channel

inv_mask = invert_mask(original_mask)
out_images = pipe(prompt=prompts, negative_prompt=negative_prompts, image=input_img, mask_image=inv_mask, guidance_image=guidance_img, **params).images

for i, image in enumerate(out_images):
    composite = Image.composite(input_img, image, original_mask)
    composite.save(f"{out_path}/{prefix}guided_inpaint_{params['strength']}_{seed}_{i}.png")