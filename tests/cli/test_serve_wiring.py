"""App wiring for `ccml serve`.

Regression guard: the stores and job managers are constructed only when the
converter actually starts, so a missing import here is invisible to every
other test and surfaces as a crash on the user's machine. `build_app` exists
to make this path reachable.
"""
from types import SimpleNamespace

import pytest

from coreml_converter.cli.commands.serve import build_app


@pytest.fixture
def config():
    return SimpleNamespace(civitai_api_key=None, output_dir="/tmp")


def test_build_app_succeeds(tmp_path, config):
    """This alone catches a NameError from a forgotten import."""
    app, notes = build_app(tmp_path, config)
    assert app is not None
    assert isinstance(notes, list)


def test_training_state_is_attached(tmp_path, config):
    app, _ = build_app(tmp_path, config)
    assert app.state.train_store is not None
    assert app.state.train_manager is not None


def test_build_state_is_still_attached(tmp_path, config):
    app, _ = build_app(tmp_path, config)
    assert app.state.build_store is not None
    assert app.state.job_manager is not None
    assert app.state.registry is not None
    assert app.state.favorites is not None


def test_interrupted_training_runs_are_failed_on_startup(tmp_path, config):
    """A converter killed mid-run must not leave the UI showing live training."""
    from coreml_converter.core.models import TrainRecord, TrainRequest, TrainStatus
    from coreml_converter.core.state import TrainStore

    store = TrainStore(tmp_path / "trainings.json")
    record = TrainRecord(
        request=TrainRequest(name="n", trigger="t", image_paths=["/a.png"],
                             output_dir=tmp_path),
        status=TrainStatus.RUNNING)
    store.save(record)

    app, notes = build_app(tmp_path, config)
    assert app.state.train_store.get(record.id).status == TrainStatus.FAILED
    assert any("training run" in n for n in notes)


def test_finished_runs_are_left_alone(tmp_path, config):
    from coreml_converter.core.models import TrainRecord, TrainRequest, TrainStatus
    from coreml_converter.core.state import TrainStore

    store = TrainStore(tmp_path / "trainings.json")
    record = TrainRecord(
        request=TrainRequest(name="n", trigger="t", image_paths=["/a.png"],
                             output_dir=tmp_path),
        status=TrainStatus.COMPLETED)
    store.save(record)

    app, _ = build_app(tmp_path, config)
    assert app.state.train_store.get(record.id).status == TrainStatus.COMPLETED
