"""Diversity pass: img2img at 0.62 preserved the hero's closeup framing too
tightly, so every variation is the same pose. Add wider/looser shots at
higher strength, where the detailed subject description (not the source
image) carries the identity.
"""
import os, sys, time, pathlib
import torch
from diffusers import (
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
from PIL import Image

CKPT = os.path.expanduser(
    "~/Library/Application Support/FannyServer/checkpoints/"
    "E1B1BF22-82DB-4957-B0F0-EC61105B54B1-URPM_v23Final.safetensors"
)
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")

SUBJECT = ("a 27 year old woman with shoulder length copper red wavy hair, "
           "green eyes, light freckles across her nose and cheeks, oval face, "
           "defined cheekbones")

WIDE = [
    ("full body shot, standing, wearing jeans and a white shirt, city sidewalk, full length photo", 0.82),
    ("waist up shot, arms crossed, wearing a black blazer, office interior", 0.80),
    ("side profile view, looking to the left, plain studio background", 0.80),
    ("medium shot from below, looking up at the sky, outdoors, dramatic clouds", 0.82),
    ("waist up, sitting on steps, wearing a hoodie, urban background", 0.80),
    ("full body, walking, wearing a long coat, autumn park, full length photo", 0.84),
]
NEG = ("lowres, blurry, deformed, disfigured, bad anatomy, extra limbs, "
       "watermark, text, cropped, worst quality, cartoon, 3d render, painting, "
       "nude, nsfw")

pipe = StableDiffusionPipeline.from_single_file(
    CKPT, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", final_sigmas_type="zero",
)
pipe = pipe.to("mps"); pipe.set_progress_bar_config(disable=True)
i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
i2i.set_progress_bar_config(disable=True)

hero = Image.open(OUT / "00.png").convert("RGB")
for i, (v, strength) in enumerate(WIDE):
    prompt = (f"photo of {SUBJECT}, {v}, high quality photograph, "
              "detailed skin texture, sharp focus")
    g = torch.Generator(device="mps").manual_seed(3000 + i)
    t = time.time()
    img = i2i(prompt=prompt, image=hero, strength=strength, negative_prompt=NEG,
              num_inference_steps=34, guidance_scale=7.5, generator=g).images[0]
    img.save(OUT / f"{16+i:02d}.png")
    print(f"[{i+1}/{len(WIDE)}] str={strength} {time.time()-t:.1f}s -> {16+i:02d}.png", flush=True)
print("diversity pass done", flush=True)
