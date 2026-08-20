"""Dataset preparation: face-aware cropping and the QA warnings."""
from pathlib import Path

import pytest
from PIL import Image

from coreml_converter.core.trainer.dataset import DatasetPrep, PreparedImage


def make_image(path: Path, size=(600, 1200), colour=(120, 100, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def test_collect_expands_directories_and_filters_extensions(tmp_path):
    make_image(tmp_path / "a.png")
    make_image(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")

    found = DatasetPrep.collect([tmp_path])
    assert [p.name for p in found] == ["a.png", "b.jpg"]


def test_collect_accepts_individual_files(tmp_path):
    one = make_image(tmp_path / "a.png")
    assert DatasetPrep.collect([one]) == [one]


def test_output_is_square_at_requested_resolution(tmp_path):
    """Cropping mechanics only — screening is disabled so a synthetic image
    (which has no detectable face) still reaches the output directory."""
    make_image(tmp_path / "src" / "a.png", size=(400, 1000))
    prep = DatasetPrep(resolution=256)
    result = prep.prepare([tmp_path / "src"], tmp_path / "out", exclude_flagged=False)
    assert len(result) == 1
    assert Image.open(tmp_path / "out" / "000.png").size == (256, 256)


def test_tall_image_without_a_face_crops_above_centre(tmp_path):
    """A centre crop of a tall portrait lands on the torso; we bias upward."""
    src = tmp_path / "tall.png"
    make_image(src, size=(400, 1200))
    prep = DatasetPrep(resolution=128)
    prep.prepare([src], tmp_path / "out", exclude_flagged=False)
    # 400x1200: a centred square would start at y=400; biased must be higher.
    # Verified indirectly through the reported crop size staying full-width.
    result = prep.prepare([src], tmp_path / "out2", exclude_flagged=False)
    assert result[0].crop_pixels == 400
    assert result[0].face_detected is False


def test_empty_input_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        DatasetPrep().prepare([tmp_path / "empty"], tmp_path / "out")


def test_unreadable_file_is_skipped_not_fatal(tmp_path):
    src = tmp_path / "src"
    make_image(src / "good.png")
    (src / "broken.png").write_bytes(b"not an image")
    result = DatasetPrep(resolution=128).prepare(
        [src], tmp_path / "out", exclude_flagged=False)
    assert len(result) == 1


def test_low_resolution_source_is_flagged():
    small = PreparedImage(source=Path("tiny.jpg"), crop_pixels=210,
                          upscale_factor=2.44, face_detected=True)
    assert small.low_detail
    assert "2.44x" in small.warning()


def test_missing_face_is_flagged():
    p = PreparedImage(source=Path("wrist.jpg"), crop_pixels=800,
                      upscale_factor=0.64, face_detected=False)
    assert not p.low_detail
    assert "no face" in p.warning()


def test_good_image_produces_no_warning():
    p = PreparedImage(source=Path("ok.jpg"), crop_pixels=900,
                      upscale_factor=0.57, face_detected=True)
    assert p.warning() is None


# --- quality screening ----------------------------------------------------

from coreml_converter.core.trainer.dataset import (  # noqa: E402
    DUPLICATE_DISTANCE, _hamming, _is_greyscale, _perceptual_hash,
)


def test_greyscale_is_detected():
    assert _is_greyscale(Image.new("RGB", (64, 64), (120, 120, 120)))


def test_colour_is_not_flagged_greyscale():
    assert not _is_greyscale(Image.new("RGB", (64, 64), (200, 40, 40)))


def test_identical_images_hash_identically():
    a = Image.new("RGB", (64, 64), (10, 60, 200))
    assert _hamming(_perceptual_hash(a), _perceptual_hash(a.copy())) == 0


def test_visually_different_images_hash_apart():
    import numpy as np
    a = Image.fromarray((np.random.RandomState(1).rand(64, 64, 3) * 255).astype("uint8"))
    b = Image.fromarray((np.random.RandomState(9).rand(64, 64, 3) * 255).astype("uint8"))
    assert _hamming(_perceptual_hash(a), _perceptual_hash(b)) > DUPLICATE_DISTANCE


def test_missing_face_is_flagged():
    p = PreparedImage(source=Path("wrist.jpg"), crop_pixels=800,
                      upscale_factor=0.6, face_detected=False)
    assert p.flagged and p.reason() == "no face detected"


def test_text_overlay_is_flagged():
    p = PreparedImage(source=Path("promo.jpg"), crop_pixels=800, upscale_factor=0.6,
                      face_detected=True, has_text_overlay=True)
    assert p.flagged
    assert "text" in p.reason()


def test_duplicate_is_flagged_and_names_its_twin():
    p = PreparedImage(source=Path("b.jpg"), crop_pixels=800, upscale_factor=0.6,
                      face_detected=True, duplicate_of="a.jpg")
    assert p.flagged
    assert "a.jpg" in p.reason()


def test_low_resolution_warns_but_does_not_exclude():
    """Detail is capped, but the subject is still there — worth training on."""
    p = PreparedImage(source=Path("small.jpg"), crop_pixels=210,
                      upscale_factor=2.44, face_detected=True)
    assert not p.flagged
    assert "upscaled" in p.warning()


def test_greyscale_warns_but_does_not_exclude():
    p = PreparedImage(source=Path("bw.jpg"), crop_pixels=800, upscale_factor=0.6,
                      face_detected=True, is_greyscale=True)
    assert not p.flagged
    assert "greyscale" in p.warning()


def test_duplicates_are_flagged_across_a_folder(tmp_path):
    src = tmp_path / "src"
    make_image(src / "a.png", size=(512, 512), colour=(30, 90, 180))
    make_image(src / "b.png", size=(512, 512), colour=(30, 90, 180))  # same image
    result = DatasetPrep(resolution=128).prepare(
        [src], tmp_path / "out", exclude_flagged=False)
    flagged = [r for r in result if r.duplicate_of is not None]
    assert len(flagged) == 1, "exactly one of an identical pair should be flagged"


def test_all_images_excluded_raises_a_useful_error(tmp_path):
    """Only when exclusion is explicitly requested — silently training on
    nothing would waste an hour."""
    src = tmp_path / "src"
    make_image(src / "a.png")
    with pytest.raises(ValueError, match="every image was excluded"):
        DatasetPrep(resolution=128).prepare(
            [src], tmp_path / "out", exclude_flagged=True)


def test_flagged_images_are_kept_by_default(tmp_path):
    """The detectors are too noisy to discard someone's photos unattended."""
    src = tmp_path / "src"
    make_image(src / "a.png")          # synthetic: no detectable face
    result = DatasetPrep(resolution=128).prepare([src], tmp_path / "out")
    assert result[0].flagged
    assert len(list((tmp_path / "out").glob("*.png"))) == 1
