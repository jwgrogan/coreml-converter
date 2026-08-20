"""Where intermediate checkpoints are written.

Regression guard: they used to be written beside the *staged* output, inside
the scratch directory that is deleted when the run ends — so every checkpoint
vanished at exactly the moment it became useful (recovering an over-trained
run, or salvaging a cancelled one). Nothing caught it because no test looked
at the destination.
"""
import inspect
from pathlib import Path

from coreml_converter.core.trainer.trainer import LoRATrainer
from coreml_converter.core.models import TrainingParams


def test_train_accepts_a_checkpoint_dir():
    sig = inspect.signature(LoRATrainer.train)
    assert "checkpoint_dir" in sig.parameters


def test_checkpoints_are_not_written_beside_the_staged_output():
    """The staged path lives in scratch; writing there is the bug."""
    source = inspect.getsource(LoRATrainer.train)
    assert "output_path.with_name(" not in source, (
        "intermediate checkpoints must not be written beside the staged output — "
        "that directory is scratch and gets deleted")
    assert "checkpoint_dir /" in source


def test_job_runner_points_checkpoints_at_the_output_directory():
    from coreml_converter.web import train_jobs
    source = inspect.getsource(train_jobs._run_training)
    assert "checkpoint_dir=output_dir" in source, (
        "the job must send intermediates to the user's LoRA directory, "
        "not the scratch dir it deletes in `finally`")


def test_scratch_is_still_cleaned_up():
    """The fix must not leak the scratch directory."""
    from coreml_converter.web import train_jobs
    source = inspect.getsource(train_jobs._run_training)
    assert "shutil.rmtree(scratch" in source


def test_checkpoint_dir_defaults_to_the_output_paths_parent():
    """Callers that do not care still get the old, sane behaviour."""
    source = inspect.getsource(LoRATrainer.train)
    assert "checkpoint_dir or output_path.parent" in source
