"""Generate a consistent-subject dataset for the LoRA training spike.

No reference photos exist on this machine and the only identity LoRA in the
library is a LyCORIS that diffusers 0.30.2 cannot load, so we bootstrap a
consistent subject the usual way: one hero portrait via txt2img, then
img2img variations off it at moderate strength, which keeps the face while
changing pose / framing / lighting / clothing.

Training on these and checking the resulting LoRA reproduces the same
subject is a self-consistency test of the trainer.
"""
import os, sys, time, pathlib
import torch
from diffusers import (
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

CKPT = os.path.expanduser(
    "~/Library/Application Support/FannyServer/checkpoints/"
    "E1B1BF22-82DB-4957-B0F0-EC61105B54B1-URPM_v23Final.safetensors"
)
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")
OUT.mkdir(parents=True, exist_ok=True)

# A specific, repeatable appearance. Detail here is what keeps the subject
# recognisable across img2img variations.
SUBJECT = ("a 27 year old woman with shoulder length copper red wavy hair, "
           "green eyes, light freckles across her nose and cheeks, oval face, "
           "defined cheekbones")

HERO = (f"photo of {SUBJECT}, closeup portrait, neutral expression, "
        "soft even studio lighting, plain grey background, "
        "high quality photograph, detailed skin texture, sharp focus, 85mm")

VARIANTS = [
    "closeup portrait, slight smile, warm window light, indoors",
    "head and shoulders, three-quarter view, natural daylight, outdoors",
    "head and shoulders, looking slightly away, overcast light, park background",
    "medium shot, wearing a white t-shirt, city street background",
    "medium shot, wearing a denim jacket, brick wall background",
    "portrait, near profile view, rim lighting, dark background",
    "portrait, laughing, golden hour sunlight, beach background",
    "closeup face, direct eye contact, ring light, studio",
    "head and shoulders, wearing a knit sweater, cozy indoor lighting",
    "medium shot, sitting at a cafe, ambient light, blurred background",
    "portrait, hair tied back, bright even lighting, white background",
    "head and shoulders, slight head tilt, evening light, balcony",
    "closeup portrait, serious expression, dramatic side lighting",
    "medium shot, wearing a summer dress, outdoor garden, sunny",
    "portrait, looking over shoulder, soft diffused light, neutral backdrop",
]

NEG = ("lowres, blurry, deformed, disfigured, bad anatomy, extra limbs, "
       "watermark, text, cropped, worst quality, cartoon, 3d render, painting, "
       "nude, nsfw")

print("loading pipeline...", flush=True)
t0 = time.time()
pipe = StableDiffusionPipeline.from_single_file(
    CKPT, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
)
# The checkpoint embeds a malformed scheduler config (algorithm_type "deis"
# with final_sigmas_type "zero"), so override both explicitly.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", final_sigmas_type="zero",
)
pipe = pipe.to("mps")
pipe.set_progress_bar_config(disable=True)
print(f"pipeline loaded in {time.time()-t0:.1f}s", flush=True)

# hero shot
g = torch.Generator(device="mps").manual_seed(7)
t = time.time()
hero = pipe(HERO, negative_prompt=NEG, num_inference_steps=30,
            guidance_scale=7.0, width=512, height=512, generator=g).images[0]
hero.save(OUT / "00.png")
print(f"[1/{len(VARIANTS)+1}] hero {time.time()-t:.1f}s -> 00.png", flush=True)

# variations off the hero, sharing its components (no second model load)
i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
i2i.set_progress_bar_config(disable=True)

times = []
for i, v in enumerate(VARIANTS):
    prompt = (f"photo of {SUBJECT}, {v}, high quality photograph, "
              "detailed skin texture, sharp focus")
    g = torch.Generator(device="mps").manual_seed(2000 + i)
    t = time.time()
    img = i2i(prompt=prompt, image=hero, strength=0.62, negative_prompt=NEG,
              num_inference_steps=32, guidance_scale=7.0, generator=g).images[0]
    dt = time.time() - t
    times.append(dt)
    img.save(OUT / f"{i+1:02d}.png")
    print(f"[{i+2}/{len(VARIANTS)+1}] {dt:.1f}s -> {i+1:02d}.png", flush=True)

print(f"\ndone. {len(times)+1} images, variations avg {sum(times)/len(times):.1f}s", flush=True)
print(f"output: {OUT.resolve()}", flush=True)
