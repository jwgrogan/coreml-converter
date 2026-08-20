"""Mode presets and the caption-strategy inversion between them."""
import pytest

from coreml_converter.core.models import (
    MODE_DEFAULTS, TrainingMode, TrainingParams, caption_for,
)


def test_character_caption_omits_any_face_description():
    """Whatever is captioned stays promptable; the face must be absorbed."""
    assert caption_for("character", "fnnyleah", "woman") == "photo of fnnyleah woman"


def test_pose_caption_describes_the_subject_not_the_pose():
    """Inverse of character: identity stays promptable, pose binds to trigger."""
    caption = caption_for("pose", "armsup", "woman")
    assert "woman" in caption and caption.endswith("armsup")


def test_style_caption_marks_the_trigger_as_a_style():
    assert caption_for("style", "grainyfilm", "woman") == "grainyfilm style"


def test_suffix_is_appended_and_normalised():
    assert caption_for("character", "t", "woman", " , windswept ") == \
        "photo of t woman, windswept"


def test_empty_suffix_leaves_caption_untouched():
    assert caption_for("character", "t", "woman", "") == "photo of t woman"


def test_character_trains_the_text_encoder_by_default():
    """F19/F20: the biggest lever for real-face identity."""
    assert TrainingParams.for_mode("character").train_text_encoder is True


def test_pose_does_not_train_the_text_encoder():
    assert TrainingParams.for_mode("pose").train_text_encoder is False


def test_style_uses_a_higher_rank_than_character():
    assert TrainingParams.for_mode("style").rank > TrainingParams.for_mode("character").rank


def test_style_targets_feed_forward_layers_too():
    style = TrainingParams.for_mode("style").resolved_targets("style")
    character = TrainingParams.for_mode("character").resolved_targets("character")
    assert len(style) > len(character)
    assert any("ff" in t for t in style)


def test_explicit_overrides_beat_mode_defaults():
    params = TrainingParams.for_mode("character", rank=32, steps=99)
    assert params.rank == 32 and params.steps == 99


def test_none_overrides_are_ignored():
    params = TrainingParams.for_mode("character", rank=None)
    assert params.rank == MODE_DEFAULTS["character"]["rank"]


def test_unknown_mode_falls_back_to_character():
    assert TrainingParams.for_mode("nonsense").rank == MODE_DEFAULTS["character"]["rank"]


@pytest.mark.parametrize("mode", [m.value for m in TrainingMode])
def test_alpha_can_always_equal_rank(mode):
    """The kohya exporter writes alpha = rank, so no preset may diverge."""
    assert TrainingParams.for_mode(mode).rank > 0


# --- style families -------------------------------------------------------

from coreml_converter.core.models import (  # noqa: E402
    RECOMMENDED_BASES, StyleFamily, recommended_base,
)


def test_every_family_has_a_recommendation():
    for family in StyleFamily:
        assert recommended_base(family.value)["name"]


def test_anime_and_photoreal_recommend_different_bases():
    """A LoRA trained on a photoreal base lands weakly on anime checkpoints."""
    assert (recommended_base("photoreal")["name"]
            != recommended_base("anime")["name"])


def test_every_family_carries_walkthrough_guidance():
    """The UI leans on these to explain the choice, so none may be blank."""
    for family in StyleFamily:
        entry = recommended_base(family.value)
        for field in ("label", "description", "hint", "dataset", "search"):
            assert entry[field], f"{family.value} missing {field}"


def test_illustration_is_distinct_from_photoreal():
    assert (recommended_base("illustration")["name"]
            != recommended_base("photoreal")["name"])


def test_unknown_family_falls_back_to_the_neutral_base():
    assert recommended_base("klingon") == RECOMMENDED_BASES["general"]


def test_each_recommendation_says_what_to_avoid():
    """The failure mode is picking a stylised merge, so name it explicitly."""
    for family in StyleFamily:
        assert recommended_base(family.value)["avoid"]


# --- the text-encoder / step-count interaction ----------------------------

from coreml_converter.core.models import (  # noqa: E402
    TE_STEP_CEILING, overtraining_warning,
)


def test_character_preset_uses_a_low_step_count_because_it_trains_the_text_encoder():
    """Measured: with TE on, quality peaked near 300 steps and 1200 was fried."""
    p = TrainingParams.for_mode("character")
    assert p.train_text_encoder is True
    assert p.steps <= TE_STEP_CEILING


def test_presets_without_text_encoder_may_use_more_steps():
    for mode in ("pose", "style"):
        p = TrainingParams.for_mode(mode)
        assert p.train_text_encoder is False
        assert p.steps >= 1200


def test_no_preset_ships_a_combination_that_overtrains():
    """Guards the exact defect found in F28 — TE on with a UNet-only step count."""
    for mode in ("character", "pose", "style"):
        p = TrainingParams.for_mode(mode)
        assert overtraining_warning(p.steps, p.train_text_encoder) is None, mode


def test_long_text_encoder_runs_are_flagged():
    warning = overtraining_warning(1500, True)
    assert warning and "overtrain" in warning


def test_long_runs_without_text_encoder_are_not_flagged():
    assert overtraining_warning(1500, False) is None


def test_the_ceiling_itself_is_not_flagged():
    assert overtraining_warning(TE_STEP_CEILING, True) is None
