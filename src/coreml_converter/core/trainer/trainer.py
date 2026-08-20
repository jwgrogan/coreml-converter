"""SD 1.5 LoRA training loop for Apple Silicon (MPS).

Shaped like core/converter/converter.py: a progress callback, a cancellation
check between steps, and periodic checkpoints so a crash costs minutes rather
than the run.

Phase 0 measurements that drive the defaults here:
  * `torch.autocast` is slower AND hungrier than plain fp32 on MPS — not used.
  * Gradient checkpointing stays on: the memory-heavy alternative collapsed to
    a third of its speed part-way through a real run on a 24GB machine.
  * VAE latents and text embeddings are pre-cached so only the UNet is in the
    training graph.
"""
from __future__ import annotations

import contextlib
import logging
import math
import time
from pathlib import Path
from typing import Callable

from coreml_converter.core.models import TrainingParams, TrainResult

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, str, int, int, int], None]
CancelFn = Callable[[], bool]


class TrainingCancelled(Exception):
    pass


class LoRATrainer:
    def __init__(self, params: TrainingParams,
                 progress: ProgressFn | None = None,
                 should_cancel: CancelFn | None = None,
                 mode: str = "character") -> None:
        self.p = params
        self.mode = mode
        self._progress = progress or (lambda *a: None)
        self._should_cancel = should_cancel or (lambda: False)

    # -- helpers ---------------------------------------------------------
    def _report(self, step: str, message: str, percent: int,
                done: int = 0, total: int = 0) -> None:
        self._progress(step, message, percent, done, total)

    def _check_cancel(self) -> None:
        if self._should_cancel():
            raise TrainingCancelled("cancelled by request")

    def _noise_scheduler(self, prediction_type: str):
        from diffusers import DDPMScheduler
        # Built explicitly: checkpoint-embedded scheduler configs can be
        # malformed (one shipped algorithm_type "deis" with
        # final_sigmas_type "zero", which from_config() rejects outright).
        return DDPMScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", clip_sample=False,
            prediction_type=prediction_type, steps_offset=1,
        )

    # -- main ------------------------------------------------------------
    def train(self, *, image_dir: Path, base_checkpoint: Path, caption: str,
              output_path: Path, metadata_fn=None,
              checkpoint_dir: Path | None = None) -> TrainResult:
        import torch
        import torch.nn.functional as F
        from diffusers import StableDiffusionPipeline
        from diffusers.training_utils import compute_snr
        from peft import LoraConfig

        from coreml_converter.core.trainer.export import export_kohya_lora

        p = self.p
        # Intermediate checkpoints must not live beside the staged output: that
        # directory is scratch and is deleted when the run ends, which would
        # destroy them at exactly the moment they become useful (recovering
        # from an over-trained run, or salvaging a cancelled one).
        checkpoint_dir = checkpoint_dir or output_path.parent
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.manual_seed(p.seed)
        t_start = time.time()

        self._report("load", "Loading base checkpoint", 2)
        # Single-file .safetensors is the common case, but plenty of models —
        # epiCRealism among them — publish only a diffusers directory layout.
        loader = (StableDiffusionPipeline.from_pretrained
                  if base_checkpoint.is_dir()
                  else StableDiffusionPipeline.from_single_file)
        pipe = loader(
            str(base_checkpoint), torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False)
        unet, vae, text_encoder = pipe.unet, pipe.vae, pipe.text_encoder
        tokenizer = pipe.tokenizer
        prediction_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
        noise_scheduler = self._noise_scheduler(prediction_type)
        self._check_cancel()

        self._report("cache", "Encoding images", 6)
        latents, vae_scale = self._cache_latents(vae, image_dir, device)
        image_count = len(latents) // (2 if p.flip_augmentation else 1)

        # Caching the caption embedding is only valid while the text encoder is
        # frozen — if we are training it, the embedding changes every step, so
        # it has to be recomputed in the loop. All captions are identical in v1,
        # so that is one short sequence per step.
        train_te = bool(p.train_text_encoder)
        embedding = None
        if not train_te:
            embedding = self._encode_caption(text_encoder, tokenizer, caption, device)

        # The VAE is never in the training graph either way.
        vae.to("cpu")
        del vae
        if not train_te:
            text_encoder.to("cpu")
            del text_encoder
        del pipe
        if device == "mps":
            torch.mps.empty_cache()
        self._check_cancel()

        self._report("prepare", "Attaching LoRA adapter", 10)
        unet.requires_grad_(False)
        # alpha must equal r — the kohya exporter writes alpha = rank regardless.
        unet.add_adapter(LoraConfig(
            r=p.rank, lora_alpha=p.rank, init_lora_weights="gaussian",
            target_modules=p.resolved_targets(self.mode)))
        if p.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if p.precision == "fp16w":
            unet.to(device, dtype=torch.float16)
            for param in unet.parameters():      # adapter stays fp32
                if param.requires_grad:
                    param.data = param.data.float()
            model_dtype = torch.float16
        else:
            unet.to(device)
            model_dtype = torch.float32
        unet.train()

        if train_te:
            text_encoder.requires_grad_(False)
            text_encoder.add_adapter(LoraConfig(
                r=p.rank, lora_alpha=p.rank, init_lora_weights="gaussian",
                target_modules=["k_proj", "q_proj", "v_proj", "out_proj"]))
            text_encoder.to(device, dtype=torch.float32)
            text_encoder.train()
            caption_ids = tokenizer(
                caption, padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt").input_ids.to(device)

        trainable = [q for q in unet.parameters() if q.requires_grad]
        if train_te:
            trainable += [q for q in text_encoder.parameters() if q.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=p.learning_rate,
                                      betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

        def lr_lambda(step: int) -> float:
            if step < p.warmup_steps:
                return step / max(1, p.warmup_steps)
            progress = (step - p.warmup_steps) / max(1, p.steps - p.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        embedding_d = embedding.to(device) if embedding is not None else None
        gen = torch.Generator(device="cpu").manual_seed(p.seed)
        losses: list[float] = []
        step_times: list[float] = []

        self._report("train", "Training", 12, 0, p.steps)
        for step in range(p.steps):
            self._check_cancel()
            tick = time.time()

            idx = torch.randint(0, len(latents), (1,), generator=gen).item()
            mean, std = latents[idx]
            latent = (mean + std * torch.randn(mean.shape, generator=gen)).unsqueeze(0)
            latent = latent.to(device) * vae_scale

            noise = torch.randn(latent.shape, generator=gen).to(device)
            timestep = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (1,), generator=gen).to(device)
            noisy = noise_scheduler.add_noise(latent, noise, timestep)
            target = (noise if prediction_type == "epsilon"
                      else noise_scheduler.get_velocity(latent, noise, timestep))

            if train_te:
                hidden = text_encoder(caption_ids)[0]
            else:
                hidden = embedding_d

            with contextlib.nullcontext():
                pred = unet(noisy.to(model_dtype), timestep,
                            encoder_hidden_states=hidden.to(model_dtype)).sample

            # Loss in fp32 regardless of weight dtype — this is what keeps the
            # fp16-weight path numerically safe.
            per_sample = F.mse_loss(pred.float(), target.float(),
                                    reduction="none").mean(dim=[1, 2, 3])
            snr = compute_snr(noise_scheduler, timestep)
            weight = torch.stack(
                [snr, p.snr_gamma * torch.ones_like(timestep)], dim=1).min(dim=1)[0] / snr
            loss = (per_sample * weight).mean() / p.grad_accum
            loss.backward()

            if (step + 1) % p.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            value = loss.item() * p.grad_accum
            if math.isnan(value) or math.isinf(value):
                raise RuntimeError(f"loss became {value} at step {step} — training diverged")
            losses.append(value)
            step_times.append(time.time() - tick)

            done = step + 1
            if done % 25 == 0 or done == p.steps:
                recent = sum(step_times[-100:]) / min(100, len(step_times))
                eta = (p.steps - done) * recent
                self._report(
                    "train",
                    f"Training — {done}/{p.steps}, about {eta / 60:.0f} min left",
                    12 + int(83 * done / p.steps), done, p.steps)

            if p.save_every and done % p.save_every == 0 and done < p.steps:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                export_kohya_lora(
                    unet, checkpoint_dir / f"{output_path.stem}-step{done}.safetensors",
                    metadata_fn(done) if metadata_fn else None,
                    text_encoder=text_encoder if train_te else None)

        self._report("export", "Writing LoRA", 96, p.steps, p.steps)
        export_kohya_lora(unet, output_path,
                          metadata_fn(p.steps) if metadata_fn else None,
                          text_encoder=text_encoder if train_te else None)

        elapsed = time.time() - t_start
        warm = step_times[5:] or step_times
        self._report("done", "Training complete", 100, p.steps, p.steps)
        return TrainResult(
            lora_path=output_path,
            steps_completed=p.steps,
            training_time=round(elapsed, 1),
            file_size_mb=round(output_path.stat().st_size / 1e6, 2),
            seconds_per_step=round(sum(warm) / len(warm), 3),
            loss_first=round(sum(losses[:50]) / min(50, len(losses)), 4),
            loss_last=round(sum(losses[-50:]) / min(50, len(losses)), 4),
            images_used=image_count,
        )

    # -- caching ---------------------------------------------------------
    def _cache_latents(self, vae, image_dir: Path, device: str):
        """Cache the latent distribution's mean/std, not a single sample, so
        each step can still draw a fresh one (kohya does the same)."""
        import numpy as np
        import torch
        from PIL import Image

        vae.to(device).eval()
        scale = float(vae.config.scaling_factor)
        paths = sorted(q for q in image_dir.iterdir() if q.suffix.lower() == ".png")
        cached = []
        with torch.no_grad():
            for path in paths:
                image = Image.open(path).convert("RGB")
                variants = [image]
                if self.p.flip_augmentation:
                    variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))
                for variant in variants:
                    array = torch.from_numpy(np.array(variant)).float()
                    tensor = (array.permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0)
                    dist = vae.encode(tensor.to(device, dtype=vae.dtype)).latent_dist
                    cached.append((dist.mean.squeeze(0).cpu().float(),
                                   dist.std.squeeze(0).cpu().float()))
        if not cached:
            raise ValueError(f"no prepared images in {image_dir}")
        return cached, scale

    def _encode_caption(self, text_encoder, tokenizer, caption: str, device: str):
        import torch
        # Every image shares one caption in v1, so this runs exactly once.
        text_encoder.to(device).eval()
        with torch.no_grad():
            ids = tokenizer(caption, padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length,
                            return_tensors="pt").input_ids.to(device)
            return text_encoder(ids)[0].cpu().float()
