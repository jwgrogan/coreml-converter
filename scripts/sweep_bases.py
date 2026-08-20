"""Sweep the training base checkpoint, holding everything else fixed.

The standing harness for "which checkpoint should we train on?" (F30/F31) and,
more generally, for any preset change: it enforces the discipline the benchmarks
doc keeps insisting on — change ONE variable, judge on fixed prompts and seeds,
never tune from a short probe.

Two phases, both resumable — rerunning skips work whose output already exists,
because a 16-minute run per base plus a generation matrix is long enough that
something will interrupt it.

  train   one training run per candidate base, sequential (the converter
          rejects concurrent runs with 409 train_busy)
  verify  generate the judgment matrix: every trained candidate on every
          generation base, fixed prompts and seeds, composed into one labeled
          sheet per generation base

Judging (see 2026-08-20-train-flow-lockdown-plan.md): a candidate is judged both
on its OWN training base (the flattering case — isolates training quality) and on
a COMMON deployment base (the transfer case — comparable across candidates, and
what the product actually promises: train once, use across the family). Primary
axis is cross-prompt identity consistency, not peak single-image quality.

Usage:
    PY=~/GitHub/coreml-converter/.venv/bin/python
    $PY scripts/sweep_bases.py --dataset ~/GitHub/fanny-server/l_dataset_curated \
        --out-root ~/GitHub/fanny-server/trained_loras/sweep \
        --trigger fnnyleah --class-token woman
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

CONVERTER = "http://127.0.0.1:8898"
CKPT_DIR = pathlib.Path(
    "~/Library/Application Support/FannyServer/checkpoints"
).expanduser()

# Candidates. `key` names the run and its output directory; `file` is relative
# to CKPT_DIR. URPM is the F31 incumbent and is included so the sheet carries its
# own control row rather than relying on a comparison generated weeks earlier.
CANDIDATES = [
    {"key": "urpm", "file": "E1B1BF22-82DB-4957-B0F0-EC61105B54B1-URPM_v23Final.safetensors"},
    {"key": "cyberrealistic", "file": "7919EC1F-A19B-44E5-A5D9-4C271BF8B2B6-cyberrealistic_final.safetensors"},
    {"key": "realisticvision", "file": "Realistic_Vision_V5.1.safetensors"},
    {"key": "perfectworld", "file": "2E2F910C-DB1B-4203-85D5-EBC95C56DB4D-perfectWorld_v6Baked.safetensors"},
]

# The common deployment base every candidate is also judged on. Not URPM: using
# one candidate's own training base as the shared yardstick would flatter it.
COMMON_BASE_KEY = "realisticvision"

# The F31 winner, held fixed across the sweep. save_every keeps the intermediate
# checkpoints that made the F29 step sweep free.
PARAMS = {
    "rank": 16,
    "steps": 400,
    "learning_rate": 1e-4,
    "resolution": 512,
    "train_text_encoder": True,
    "precision": "fp32",
    "gradient_checkpointing": True,
    "save_every": 100,
    "seed": 42,
}

# Which saved checkpoint to judge. F31 put the peak at 300 with 400 as tiebreak.
JUDGE_STEPS = [300, 400]


def _post(path, body):
    req = urllib.request.Request(
        CONVERTER + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r), 200
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}"), e.code


def _get(path):
    with urllib.request.urlopen(CONVERTER + path) as r:
        return json.load(r)


def require_converter():
    """Fail early and say how to fix it.

    The converter is a separate process with its own lifecycle; it being down is
    the single most likely reason this script cannot start, and a raw
    ConnectionRefusedError traceback buries that.
    """
    try:
        h = _get("/api/health")
    except Exception:
        raise SystemExit(
            "converter not reachable at " + CONVERTER + "\n"
            "start it with:  ~/GitHub/coreml-converter/.venv/bin/ccml serve --port 8898"
        )
    if not h.get("ml_deps_ok", True):
        raise SystemExit(f"converter is up but missing deps: {h.get('missing_deps')}")


def checkpoint_path(key):
    for c in CANDIDATES:
        if c["key"] == key:
            return CKPT_DIR / c["file"]
    raise SystemExit(f"unknown checkpoint key: {key}")


def lora_file(out_root, key, step):
    """Where the converter leaves a given candidate's checkpoint.

    Intermediate checkpoints are suffixed; the final one is not — a difference
    that has bitten before, so resolve it in one place.
    """
    d = out_root / f"exp_{key}"
    name = f"leah-{key}-te400"
    final = d / f"{name}.safetensors"
    if step >= PARAMS["steps"]:
        return final
    return d / f"{name}-step{step}.safetensors"


def train_all(args, out_root):
    images = sorted(
        p for p in pathlib.Path(args.dataset).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if len(images) < 10:
        raise SystemExit(f"{args.dataset}: {len(images)} images; training needs >= 10")
    print(f"dataset: {len(images)} images from {args.dataset}\n")

    for cand in CANDIDATES:
        key = cand["key"]
        outdir = out_root / f"exp_{key}"
        # Resume: a finished run leaves the final file behind.
        if lora_file(out_root, key, PARAMS["steps"]).exists():
            print(f"[{key}] already trained -> skipping")
            continue
        base = checkpoint_path(key)
        if not base.exists():
            print(f"[{key}] MISSING checkpoint {base} -> skipping", file=sys.stderr)
            continue

        body = {
            "name": f"leah-{key}-te400",
            "trigger": args.trigger,
            "mode": "character",
            "style_family": "photoreal",
            "class_token": args.class_token,
            "image_paths": [str(p) for p in images],
            "base_path": str(base),
            "output_dir": str(outdir),
            "params": PARAMS,
        }
        resp, code = _post("/api/train/start", body)
        if code == 409:
            raise SystemExit("converter busy with another run; wait for it and rerun")
        if code != 200:
            print(f"[{key}] start failed ({code}): {resp}", file=sys.stderr)
            continue

        train_id = resp["train_id"]
        print(f"[{key}] training {train_id} ({resp['steps']} steps)")
        started = time.time()
        while True:
            time.sleep(30)
            try:
                s = _get(f"/api/train/{train_id}/status")
            except Exception as e:  # transient; the run outlives a blip
                print(f"  (status unavailable: {e})")
                continue
            if s["status"] in ("completed", "failed", "cancelled"):
                mins = (time.time() - started) / 60
                print(f"[{key}] {s['status']} in {mins:.1f} min")
                if s["status"] == "completed":
                    r = s.get("result") or {}
                    print(f"       {r.get('seconds_per_step')} s/step, "
                          f"loss {r.get('loss_first')} -> {r.get('loss_last')}")
                else:
                    print(f"       error: {s.get('error')}", file=sys.stderr)
                break
            print(f"  {s.get('message', s['status'])}")


def verify_all(args, out_root):
    """Generate every (candidate LoRA) x (generation base) cell."""
    here = pathlib.Path(__file__).parent
    sheets = out_root / "verify"

    for step in JUDGE_STEPS:
        for cand in CANDIDATES:
            key = cand["key"]
            lora = lora_file(out_root, key, step)
            if not lora.exists():
                continue
            # Own base (flattering case) + the common deployment base
            # (transfer case). Same LoRA, two generation checkpoints.
            for gen_key in {key, COMMON_BASE_KEY}:
                out = sheets / f"step{step}" / f"gen_{gen_key}" / key
                if (out / "lora_3.png").exists():
                    print(f"[{key} step{step} on {gen_key}] exists -> skipping")
                    continue
                out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, str(here / "verify_lora.py"),
                    "--lora", str(lora), "--out", str(out),
                    "--trigger", args.trigger, "--class-token", args.class_token,
                    "--base", str(checkpoint_path(gen_key)),
                    "--skip-base-row",
                ]
                print(f"[{key} step{step} on {gen_key}] generating")
                subprocess.run(cmd, check=True)

    compose(args, out_root, sheets)


def compose(args, out_root, sheets):
    """One labeled sheet per (step, generation base): rows are training bases."""
    from PIL import Image, ImageDraw

    W, LABEL, NPROMPT = 512, 40, 4
    for step in JUDGE_STEPS:
        for gen_key in {c["key"] for c in CANDIDATES} | {COMMON_BASE_KEY}:
            rows = []
            for cand in CANDIDATES:
                d = sheets / f"step{step}" / f"gen_{gen_key}" / cand["key"]
                if (d / "lora_0.png").exists():
                    rows.append((cand["key"], d))
            if len(rows) < 2:
                continue
            sheet = Image.new("RGB", (W * NPROMPT, (W + LABEL) * len(rows)), "white")
            d = ImageDraw.Draw(sheet)
            for r, (label, src) in enumerate(rows):
                y = r * (W + LABEL)
                tag = f"trained on {label}" + ("  (own base)" if label == gen_key else "")
                d.text((10, y + 12), tag, fill="black")
                for c in range(NPROMPT):
                    sheet.paste(Image.open(src / f"lora_{c}.png"), (c * W, y + LABEL))
            out = sheets / f"judgment_step{step}_on_{gen_key}.png"
            sheet.save(out)
            print(f"sheet -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--trigger", default="fnnyleah")
    ap.add_argument("--class-token", default="woman")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    # Runs for over an hour and is normally watched through a redirected log;
    # block buffering would make it look hung.
    sys.stdout.reconfigure(line_buffering=True)

    out_root = pathlib.Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        require_converter()
        train_all(args, out_root)
    if not args.skip_verify:
        verify_all(args, out_root)


if __name__ == "__main__":
    main()
