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
