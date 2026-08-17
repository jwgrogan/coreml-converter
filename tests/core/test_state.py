import pytest
from pathlib import Path
from coreml_converter.core.state import BuildStore
from coreml_converter.core.models import (
    BuildRecord, Recipe, ModelInfo, ModelSource, BaseArchitecture,
    ModelType, ConversionConfig, BuildStatus,
)


def _make_recipe(name: str = "test") -> Recipe:
    base = ModelInfo(
        source=ModelSource.CIVITAI, id="1", name="Base",
        base_architecture=BaseArchitecture.SD15,
        model_type=ModelType.CHECKPOINT, tags=[], download_url="", metadata={},
    )
    config = ConversionConfig(output_dir=Path("/tmp"), model_name=name)
    return Recipe(name=name, base_model=base, loras=[], conversion_config=config)


class TestBuildStore:
    def test_create_and_get(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        record = BuildRecord(recipe=_make_recipe())
        store.save(record)
        loaded = store.get(record.id)
        assert loaded is not None
        assert loaded.id == record.id
        assert loaded.status == BuildStatus.PENDING

    def test_list_all(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        store.save(BuildRecord(recipe=_make_recipe("a")))
        store.save(BuildRecord(recipe=_make_recipe("b")))
        records = store.list_all()
        assert len(records) == 2

    def test_update_status(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        record = BuildRecord(recipe=_make_recipe())
        store.save(record)
        record.status = BuildStatus.RUNNING
        store.save(record)
        loaded = store.get(record.id)
        assert loaded.status == BuildStatus.RUNNING

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = BuildStore(tmp_path / "builds.json")
        assert store.get("nonexistent") is None

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "builds.json"
        store1 = BuildStore(path)
        record = BuildRecord(recipe=_make_recipe())
        store1.save(record)
        store2 = BuildStore(path)
        loaded = store2.get(record.id)
        assert loaded is not None


class TestFailInterrupted:
    """Builds run in-process; a record still marked RUNNING at startup belongs
    to a process that is gone, and would otherwise show as running forever."""

    def _record(self, tmp_path, status):
        from coreml_converter.core.models import (
            BaseArchitecture, BuildRecord, ConversionConfig, ModelInfo,
            ModelSource, ModelType, Recipe,
        )
        return BuildRecord(
            recipe=Recipe(
                name="m",
                base_model=ModelInfo(
                    source=ModelSource.CIVITAI, id="1", name="B",
                    base_architecture=BaseArchitecture.SD15,
                    model_type=ModelType.CHECKPOINT, tags=[],
                    download_url="", metadata={},
                ),
                loras=[],
                conversion_config=ConversionConfig(output_dir=tmp_path, model_name="m"),
            ),
            status=status,
        )

    def test_running_builds_become_failed(self, tmp_path):
        from coreml_converter.core.models import BuildStatus
        from coreml_converter.core.state import BuildStore

        store = BuildStore(tmp_path / "builds.json")
        record = self._record(tmp_path, BuildStatus.RUNNING)
        store.save(record)

        assert store.fail_interrupted() == 1

        reloaded = store.get(record.id)
        assert reloaded.status == BuildStatus.FAILED
        assert "Interrupted" in reloaded.error
        assert reloaded.completed_at is not None

    def test_finished_builds_are_untouched(self, tmp_path):
        from coreml_converter.core.models import BuildStatus
        from coreml_converter.core.state import BuildStore

        store = BuildStore(tmp_path / "builds.json")
        done = self._record(tmp_path, BuildStatus.COMPLETED)
        failed = self._record(tmp_path, BuildStatus.FAILED)
        store.save(done)
        store.save(failed)

        assert store.fail_interrupted() == 0
        assert store.get(done.id).status == BuildStatus.COMPLETED
        assert store.get(failed.id).error is None

    def test_empty_store_is_fine(self, tmp_path):
        from coreml_converter.core.state import BuildStore
        assert BuildStore(tmp_path / "builds.json").fail_interrupted() == 0
