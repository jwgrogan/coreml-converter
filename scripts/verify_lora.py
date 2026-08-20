"""Verify a trained LoRA loads through the converter's merge path and
actually reproduces the trained subject.

Uses exactly the calls core/merger/merger.py uses (load_lora_weights +
fuse_lora), so a pass here means the exported file is merge-compatible.
Generates matched with/without-LoRA pairs for a like-for-like comparison.
"""
import os, sys, time, pathlib, argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

CKPT = os.path.expanduser(
    "~/Library/Application Support/FannyServer/checkpoints/"
    "E1B1BF22-82DB-4957-B0F0-EC61105B54B1-URPM_v23Final.safetensors"
)

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--out", default="verify")
ap.add_argument("--trigger", default="fnnyspike")
ap.add_argument("--class-token", default="woman")
ap.add_argument("--scale", type=float, default=0.8)
ap.add_argument("--steps", type=int, default=28)
args = ap.parse_args()

OUT = pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
subj = f"{args.trigger} {args.class_token}"

PROMPTS = [
    f"photo of {subj}, closeup portrait, neutral expression, studio lighting, plain background",
    f"photo of {subj}, head and shoulders, outdoors, natural daylight",
    f"photo of {subj}, medium shot, wearing a red jacket, city street",
    f"photo of {subj}, portrait, smiling, golden hour sunlight",
]
NEG = ("lowres, blurry, deformed, disfigured, bad anatomy, watermark, text, "
       "cropped, worst quality, cartoon, 3d render, nude, nsfw")

print("loading pipeline...", flush=True)
pipe = StableDiffusionPipeline.from_single_file(
    CKPT, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", final_sigmas_type="zero",
)
pipe = pipe.to("mps")
pipe.set_progress_bar_config(disable=True)


def run(tag):
    imgs = []
    for i, p in enumerate(PROMPTS):
        g = torch.Generator(device="mps").manual_seed(500 + i)
        img = pipe(p, negative_prompt=NEG, num_inference_steps=args.steps,
                   guidance_scale=7.0, width=512, height=512, generator=g).images[0]
        img.save(OUT / f"{tag}_{i}.png")
        imgs.append(img)
        print(f"  {tag} {i}", flush=True)
    return imgs


print("== baseline (no lora) ==", flush=True)
base = run("base")

print(f"== loading lora {args.lora} (merger code path) ==", flush=True)
lf = pathlib.Path(args.lora)
t = time.time()
pipe.load_lora_weights(str(lf.parent), weight_name=lf.name)
pipe.fuse_lora(lora_scale=args.scale)
pipe.unload_lora_weights()
print(f"   loaded + fused in {time.time()-t:.1f}s  <-- merge compatibility OK", flush=True)

print("== with lora ==", flush=True)
lora = run("lora")

# side-by-side contact sheet: baseline row over lora row
W = H = 512
sheet = Image.new("RGB", (W * len(PROMPTS), H * 2), "white")
for i, im in enumerate(base):
    sheet.paste(im, (i * W, 0))
for i, im in enumerate(lora):
    sheet.paste(im, (i * W, H))
sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS)
sheet.save(OUT / "comparison.png")
print(f"\ncontact sheet -> {OUT/'comparison.png'} (top row: base, bottom row: +lora)", flush=True)
