"""Phase 0 de-risk spike: SD 1.5 LoRA training on Apple Silicon MPS.

Structured to mirror the module split the plan calls for in Phase 1
(dataset prep / training loop / kohya export) so it can be lifted into
coreml-converter's core/trainer/ with minimal rework.

Usage:
  spike_train_lora.py --data dataset --out out --max-steps 1500
  spike_train_lora.py --data dataset --max-steps 20 --precision fp32 --bench
"""
from __future__ import annotations
import argparse, contextlib, json, math, os, time, pathlib, sys
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.training_utils import compute_snr
from diffusers.utils.state_dict_utils import convert_state_dict_to_kohya
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

DEFAULT_CKPT = os.path.expanduser(
    "~/Library/Application Support/FannyServer/checkpoints/"
    "E1B1BF22-82DB-4957-B0F0-EC61105B54B1-URPM_v23Final.safetensors"
)


# ---------------------------------------------------------------- dataset ---
def load_images(data_dir: pathlib.Path, resolution: int) -> list[Image.Image]:
    paths = sorted(
        p for p in data_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not paths:
        raise SystemExit(f"no images found in {data_dir}")
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        # center-crop to square, then resize to target
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)
        out.append(img)
    return out


def to_tensor(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(
        __import__("numpy").array(img)
    ).float().permute(2, 0, 1) / 127.5 - 1.0
    return t


@torch.no_grad()
def cache_latents(vae, images, device, flip_aug=True):
    """Encode once up front so the VAE is out of the training graph.

    Caches the latent distribution's mean/std rather than a single sample,
    so each training step can still draw a fresh sample (kohya does the
    same); with flip_aug the mirrored image is cached as a second entry.
    """
    cached = []
    for img in images:
        variants = [img, img.transpose(Image.FLIP_LEFT_RIGHT)] if flip_aug else [img]
        for v in variants:
            x = to_tensor(v).unsqueeze(0).to(device, dtype=vae.dtype)
            dist = vae.encode(x).latent_dist
            cached.append((dist.mean.squeeze(0).cpu().float(),
                           dist.std.squeeze(0).cpu().float()))
    return cached


@torch.no_grad()
def encode_caption(text_encoder, tokenizer, caption, device):
    ids = tokenizer(
        caption, padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(ids)[0].cpu().float()


# ----------------------------------------------------------------- export ---
def export_kohya(unet, path: pathlib.Path, meta: dict):
    peft_sd = get_peft_model_state_dict(unet)
    # The "unet." prefix is mandatory: convert_state_dict_to_kohya only emits
    # the `lora_unet_` header when it sees "unet" in the key. Without it the
    # file loads nowhere. Note the converter also writes alpha = rank
    # unconditionally, so lora_alpha must equal r or the exported alpha lies.
    peft_sd = {f"unet.{k}": v for k, v in peft_sd.items()}
    kohya_sd = convert_state_dict_to_kohya(peft_sd)
    kohya_sd = {k: v.to(torch.float16).contiguous() for k, v in kohya_sd.items()}
    # safetensors metadata must be str->str; ss_* keys are the kohya
    # convention that A1111 / Draw Things / our own merger read.
    md = {k: str(v) for k, v in meta.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(kohya_sd, str(path), metadata=md)
    return len(kohya_sd)


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="out")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--trigger", default="fnnyspike")
    ap.add_argument("--class-token", default="woman")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--snr-gamma", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=250)
    # amp   = fp32 weights + autocast fp16 ops (memory-safe, bandwidth-heavy)
    # fp32  = everything fp32 (reference)
    # fp16w = fp16 base weights, fp32 LoRA params (peft casts around the
    #         adapter and restores the base dtype, so this stays coherent)
    ap.add_argument("--precision", choices=["amp", "fp32", "fp16w"], default="amp")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="skip gradient checkpointing (faster; costs memory)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench", action="store_true", help="benchmark only, skip export")
    args = ap.parse_args()

    device = "mps"
    torch.manual_seed(args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = f"photo of {args.trigger} {args.class_token}"

    print(f"== loading checkpoint (fp32 unet) ...", flush=True)
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_single_file(
        args.ckpt, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    load_s = time.time() - t0
    print(f"   loaded in {load_s:.1f}s", flush=True)

    unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")

    # Build the DDPM scheduler explicitly: the checkpoint's embedded
    # scheduler config is malformed (deis + final_sigmas_type=zero), so
    # from_config() on it raises.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        prediction_type=pred_type, steps_offset=1,
    )
    print(f"   prediction_type={pred_type}", flush=True)

    # --- cache latents + embeddings, then drop VAE/TE entirely -------------
    images = load_images(pathlib.Path(args.data), args.resolution)
    print(f"== caching {len(images)} images (+flip) ...", flush=True)
    vae.to(device).eval()
    te.to(device).eval()
    t0 = time.time()
    latents = cache_latents(vae, images, device)
    emb = encode_caption(te, tok, caption, device)
    vae_scale = float(vae.config.scaling_factor)
    print(f"   cached {len(latents)} latents + 1 embedding in {time.time()-t0:.1f}s", flush=True)
    print(f"   caption: {caption!r}", flush=True)
    vae.to("cpu"); te.to("cpu")
    del vae, te, pipe
    torch.mps.empty_cache()

    # --- attach LoRA -------------------------------------------------------
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    if not args.no_grad_ckpt:
        unet.enable_gradient_checkpointing()
    if args.precision == "fp16w":
        unet.to(device, dtype=torch.float16)
        # keep the trainable adapter in fp32 so optimizer math stays stable
        for p in unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    else:
        unet.to(device)
    unet.train()
    model_dtype = torch.float16 if args.precision == "fp16w" else torch.float32

    params = [p for p in unet.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"== lora rank {args.rank}: {n_params/1e6:.2f}M trainable params "
          f"({len(params)} tensors)", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=1e-2, eps=1e-8)

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = args.precision == "amp"
    print(f"== training {args.max_steps} steps, precision={args.precision}, "
          f"grad_accum={args.grad_accum}, lr={args.lr}", flush=True)

    emb_d = emb.to(device)
    step_times, losses, loss_win = [], [], []
    peak_mem = 0.0
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t_start = time.time()

    for step in range(args.max_steps):
        ts = time.time()
        idx = torch.randint(0, len(latents), (1,), generator=g).item()
        mean, std = latents[idx]
        lat = (mean + std * torch.randn(mean.shape, generator=g)).unsqueeze(0).to(device)
        lat = lat * vae_scale

        noise = torch.randn(lat.shape, generator=g).to(device)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                          (1,), generator=g).to(device)
        noisy = noise_scheduler.add_noise(lat, noise, t)
        target = noise if pred_type == "epsilon" else \
            noise_scheduler.get_velocity(lat, noise, t)

        ctx = torch.autocast(device_type="mps", dtype=torch.float16) if use_amp \
            else contextlib.nullcontext()
        with ctx:
            # .to() is a no-op unless fp16w put the base weights in fp16
            pred = unet(noisy.to(model_dtype), t,
                        encoder_hidden_states=emb_d.to(model_dtype)).sample

        # loss in fp32 regardless of autocast: MPS fp16 reductions are the
        # main numerical-stability risk this spike is testing.
        per = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
        snr = compute_snr(noise_scheduler, t)
        w = torch.stack([snr, args.snr_gamma * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
        loss = (per * w).mean() / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        l = loss.item() * args.grad_accum
        if math.isnan(l) or math.isinf(l):
            print(f"!! NaN/Inf loss at step {step} — aborting", flush=True)
            sys.exit(3)
        losses.append(l); loss_win.append(l)
        step_times.append(time.time() - ts)

        if (step + 1) % 25 == 0:
            recent = sum(step_times[-25:]) / 25
            avg_loss = sum(loss_win) / len(loss_win); loss_win = []
            done = step + 1
            eta = (args.max_steps - done) * (sum(step_times[-100:]) / min(100, len(step_times)))
            mem = torch.mps.driver_allocated_memory() / 1e9
            peak_mem = max(peak_mem, mem)
            print(f"   step {done}/{args.max_steps}  loss {avg_loss:.4f}  "
                  f"{recent:.2f}s/step  mem {mem:.1f}GB  eta {eta/60:.1f}m", flush=True)

        if not args.bench and args.save_every and (step + 1) % args.save_every == 0 \
                and (step + 1) < args.max_steps:
            export_kohya(unet, out_dir / f"{args.trigger}-step{step+1}.safetensors",
                         {"ss_network_dim": args.rank, "ss_network_alpha": args.rank,
                          "ss_steps": step + 1})
            print(f"   [checkpoint saved at step {step+1}]", flush=True)

    total = time.time() - t_start
    n = len(step_times)
    warm = step_times[5:] or step_times
    stats = {
        "steps": n,
        "total_seconds": round(total, 1),
        "s_per_step_mean": round(sum(warm) / len(warm), 3),
        "s_per_step_first5": round(sum(step_times[:5]) / min(5, n), 3),
        "peak_driver_mem_gb": round(max(peak_mem, torch.mps.driver_allocated_memory() / 1e9), 2),
        "loss_first50": round(sum(losses[:50]) / min(50, n), 4),
        "loss_last50": round(sum(losses[-50:]) / min(50, n), 4),
        "precision": args.precision,
        "grad_accum": args.grad_accum,
        "rank": args.rank,
        "lr": args.lr,
        "trainable_params_m": round(n_params / 1e6, 2),
        "checkpoint_load_seconds": round(load_s, 1),
    }
    print("\n== stats ==", flush=True)
    print(json.dumps(stats, indent=2), flush=True)

    if args.bench:
        (out_dir / f"bench-{args.precision}.json").write_text(json.dumps(stats, indent=2))
        return

    final = out_dir / f"{args.trigger}.safetensors"
    n_t = export_kohya(unet, final, {
        "ss_network_dim": args.rank,
        "ss_network_alpha": args.rank,
        "ss_steps": args.max_steps,
        "ss_learning_rate": args.lr,
        "ss_resolution": f"({args.resolution}, {args.resolution})",
        "ss_num_train_images": len(images),
        "fanny_trigger": args.trigger,
        "fanny_class_token": args.class_token,
        "fanny_caption": caption,
        "fanny_base_checkpoint": pathlib.Path(args.ckpt).name,
    })
    stats["output"] = str(final)
    stats["output_tensors"] = n_t
    stats["output_mb"] = round(final.stat().st_size / 1e6, 2)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nexported {n_t} tensors -> {final} ({stats['output_mb']} MB)", flush=True)


if __name__ == "__main__":
    main()
