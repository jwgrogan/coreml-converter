"""Face-aware square crop for real-world reference photos.

Real user datasets are tall phone/web images (this one averages a 0.55
aspect ratio), where a naive centre crop lands on the torso rather than the
face. Detect the face, crop a head-and-shoulders square around it, and fall
back to a top-biased crop when detection fails.
"""
import sys, pathlib
import cv2
import numpy as np
from PIL import Image

SRC = pathlib.Path(sys.argv[1])
DST = pathlib.Path(sys.argv[2])
RES = int(sys.argv[3]) if len(sys.argv) > 3 else 512
DST.mkdir(parents=True, exist_ok=True)

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

exts = {".jpg", ".jpeg", ".png", ".webp"}
paths = sorted(p for p in SRC.iterdir() if p.suffix.lower() in exts)
print(f"{len(paths)} source images -> {DST} @ {RES}px\n")

kept, notes = 0, []
for i, p in enumerate(paths):
    im = Image.open(p).convert("RGB")
    W, H = im.size
    arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6,
                                     minSize=(40, 40))

    if len(faces):
        # largest face; expand to head + shoulders
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = fx + fw / 2, fy + fh / 2
        side = min(max(fw, fh) * 3.0, min(W, H))
        cy -= side * 0.06          # a little headroom above the eyeline
        how = f"face {fw}x{fh}"
    else:
        # no face: tall images put the subject up top, so bias the crop up
        side = min(W, H)
        cx = W / 2
        cy = side / 2 + (H - side) * 0.18 if H > W else H / 2
        how = "NO FACE (top-biased)"

    left = int(round(min(max(cx - side / 2, 0), W - side)))
    top = int(round(min(max(cy - side / 2, 0), H - side)))
    side = int(round(side))
    crop = im.crop((left, top, left + side, top + side))
    up = RES / side
    crop = crop.resize((RES, RES), Image.LANCZOS)
    crop.save(DST / f"{i:02d}.png")
    kept += 1
    flag = "  <-- UPSCALED" if up > 1.35 else ""
    print(f"[{i:02d}] {W}x{H} -> crop {side}px (x{up:.2f}) {how}{flag}   {p.name[:40]}")
    if up > 1.35:
        notes.append((p.name, side, up))
    if "NO FACE" in how:
        notes.append((p.name, side, 0))

print(f"\nprepared {kept} images")
if notes:
    print("review:")
    for n, s, u in notes:
        print(f"  - {n[:48]}: crop {s}px" + (f", upscaled x{u:.2f}" if u else ", no face detected"))
