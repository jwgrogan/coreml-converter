"""Turn a folder of reference photos into training-ready square crops.

Real user datasets are tall phone and web images — the first real set averaged
a 0.55 aspect ratio — so a naive centre crop lands on the torso rather than the
face. We detect the face and crop a head-and-shoulders square around it,
falling back to a top-biased crop when detection fails.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Below this the crop is mostly interpolation; fine facial detail is simply not
# in the source and no amount of training recovers it.
UPSCALE_WARN_THRESHOLD = 1.35


@dataclass
class PreparedImage:
    source: Path
    crop_pixels: int
    upscale_factor: float
    face_detected: bool
    has_text_overlay: bool = False
    is_greyscale: bool = False
    duplicate_of: str | None = None
    source_hash: int = 0

    @property
    def low_detail(self) -> bool:
        return self.upscale_factor > UPSCALE_WARN_THRESHOLD

    @property
    def flagged(self) -> bool:
        """Worth a human look before training on it.

        Deliberately *advisory*. Measured against a real 14-image set, the Haar
        face detector produced three false negatives out of four (it misses
        non-frontal, greyscale and low-resolution faces) and the text heuristic
        both false-positived and false-negatived once. Auto-excluding on
        signals this noisy would silently discard good photos and keep bad
        ones, so the caller decides — see `exclude_flagged`.
        """
        return (not self.face_detected
                or self.has_text_overlay
                or self.duplicate_of is not None)

    def reason(self) -> str | None:
        """Why this image was excluded, phrased for the user."""
        if not self.face_detected:
            return "no face detected"
        if self.has_text_overlay:
            return "text or graphics burned into the image"
        if self.duplicate_of is not None:
            return f"near-duplicate of {self.duplicate_of}"
        return None

    def warning(self) -> str | None:
        notes = []
        if (r := self.reason()):
            notes.append(r)
        if self.low_detail:
            notes.append(f"upscaled {self.upscale_factor:.2f}x from a "
                         f"{self.crop_pixels}px crop")
        if self.is_greyscale:
            notes.append("greyscale — will bias colour output if the rest are colour")
        if not notes:
            return None
        return f"{self.source.name}: " + ", ".join(notes)


def _detect_text_overlay(image) -> bool:
    """Heuristic for burned-in captions and graphics.

    Real photographs have smoothly varying edge density; rendered text puts a
    dense band of high-contrast horizontal edges into one strip of the image.
    We look for a row-band whose edge density is far above the image median.
    This is deliberately conservative — a false exclusion costs a usable photo,
    so the threshold is set to catch obvious caption bars, not subtle marks.
    """
    import cv2
    import numpy as np

    grey = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grey, 100, 200)
    height = edges.shape[0]
    band = max(8, height // 16)
    densities = [edges[i:i + band].mean() for i in range(0, height - band, band)]
    if len(densities) < 4:
        return False
    densities = np.array(densities)
    median = float(np.median(densities))
    peak = float(densities.max())
    # A caption bar is both dense in absolute terms and far above the rest.
    return peak > 28.0 and median > 0 and peak > median * 3.2


def _is_greyscale(image) -> bool:
    """True when the image carries essentially no colour."""
    import numpy as np
    arr = np.array(image).astype("int16")
    if arr.ndim != 3 or arr.shape[2] < 3:
        return True
    spread = np.abs(arr[:, :, 0] - arr[:, :, 1]) + np.abs(arr[:, :, 1] - arr[:, :, 2])
    return float(spread.mean()) < 8.0


def _perceptual_hash(image) -> int:
    """64-bit average hash, for spotting near-duplicate frames."""
    import numpy as np
    small = image.convert("L").resize((8, 8))
    pixels = np.array(small, dtype="float32")
    bits = pixels > pixels.mean()
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# Two images within this Hamming distance are the same moment, not variety.
DUPLICATE_DISTANCE = 6


class DatasetPrep:
    """Face-aware square cropping and quality screening for reference photos."""

    def __init__(self, resolution: int = 512) -> None:
        self.resolution = resolution
        self._cascade = None

    def _face_cascade(self):
        if self._cascade is None:
            import cv2
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return self._cascade

    @staticmethod
    def collect(paths: list[str] | list[Path]) -> list[Path]:
        """Expand a mix of files and directories into a sorted image list."""
        out: list[Path] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                out.extend(sorted(c for c in p.iterdir()
                                  if c.suffix.lower() in IMAGE_EXTENSIONS))
            elif p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
                out.append(p)
        return out

    def _detect_face(self, image):
        import cv2
        import numpy as np
        arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade().detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=6, minSize=(40, 40))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def prepare_one(self, path: Path, out_path: Path) -> PreparedImage:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        width, height = image.size
        face = self._detect_face(image)

        if face is not None:
            fx, fy, fw, fh = face
            cx, cy = fx + fw / 2, fy + fh / 2
            side = min(max(fw, fh) * 3.0, min(width, height))
            cy -= side * 0.06  # a little headroom above the eyeline
            detected = True
        else:
            # Tall images put the subject up top, so bias the crop upward
            # rather than taking the middle (which is usually torso).
            side = min(width, height)
            cx = width / 2
            cy = side / 2 + (height - side) * 0.18 if height > width else height / 2
            detected = False

        side_i = int(round(side))
        left = int(round(min(max(cx - side / 2, 0), width - side)))
        top = int(round(min(max(cy - side / 2, 0), height - side)))
        crop = image.crop((left, top, left + side_i, top + side_i))
        crop = crop.resize((self.resolution, self.resolution), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path)

        return PreparedImage(
            source=path,
            crop_pixels=side_i,
            upscale_factor=self.resolution / side_i,
            face_detected=detected,
            # Screen the original, not the crop: a caption bar along the bottom
            # of a tall photo may sit outside the face crop yet still indicate
            # the kind of image we do not want in a dataset.
            has_text_overlay=_detect_text_overlay(image),
            is_greyscale=_is_greyscale(crop),
            # Hash the *original*: every face crop of one person looks alike at
            # 8x8, so hashing crops flags unrelated photos as duplicates.
            source_hash=_perceptual_hash(image),
        )

    def prepare(self, paths: list[str] | list[Path], out_dir: Path,
                exclude_flagged: bool = False) -> list[PreparedImage]:
        """Crop and screen every image, writing the usable ones to `out_dir`.

        Returns *all* prepared images with their flags, so the UI can show the
        user what looks questionable. Flagged images are still written unless
        `exclude_flagged` is set, because the detectors are not reliable enough
        to discard someone's photos unattended (see `PreparedImage.flagged`).
        """
        from PIL import Image

        sources = self.collect(paths)
        if not sources:
            raise ValueError("no usable images found (need .png/.jpg/.jpeg/.webp)")
        out_dir.mkdir(parents=True, exist_ok=True)

        staging = out_dir / "_staging"
        staging.mkdir(parents=True, exist_ok=True)
        # Pair each result with the file it produced: a source that fails to
        # load leaves a gap, so positional indices cannot be recomputed later.
        staged_pairs: list[tuple[Path, PreparedImage]] = []
        for i, src in enumerate(sources):
            staged = staging / f"{i:03d}.png"
            try:
                staged_pairs.append((staged, self.prepare_one(src, staged)))
            except Exception as exc:      # a single unreadable file must not kill the run
                logger.warning("skipping %s: %s", src, exc)
        if not staged_pairs:
            raise ValueError("every image failed to load")
        prepared = [item for _, item in staged_pairs]

        self._flag_duplicates(prepared)

        kept = 0
        for staged, item in staged_pairs:
            if not staged.exists():
                continue
            if exclude_flagged and item.flagged:
                staged.unlink(missing_ok=True)
                continue
            staged.rename(out_dir / f"{kept:03d}.png")
            kept += 1
        shutil.rmtree(staging, ignore_errors=True)

        for item in prepared:
            if (w := item.warning()):
                logger.warning("dataset: %s", w)
        if kept == 0:
            raise ValueError(
                "every image was excluded — check the warnings above; "
                "training needs photos with a clearly visible face and no burned-in text")
        return prepared

    @staticmethod
    def _flag_duplicates(prepared: list["PreparedImage"]) -> None:
        """Mark near-identical source photos, keeping the first of each group."""
        hashes: list[tuple[int, PreparedImage]] = []
        for item in prepared:
            for seen, original in hashes:
                if _hamming(item.source_hash, seen) <= DUPLICATE_DISTANCE:
                    item.duplicate_of = original.source.name
                    break
            else:
                hashes.append((item.source_hash, item))
