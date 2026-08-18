# tests/web/test_api_routes.py
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from coreml_converter.core.models import (
    BaseArchitecture,
    BuildRecord,
    BuildStatus,
    ConversionConfig,
    ModelInfo,
    ModelSource,
    ModelType,
    Recipe,
)
from coreml_converter.core.state import BuildStore
from coreml_converter.web import uploads
from coreml_converter.web.app import create_app


@pytest.fixture(autouse=True)
def clear_uploads():
    uploads.clear()
    yield
    uploads.clear()


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.search.return_value = []
    registry.get_by_id.return_value = None
    return registry


@pytest.fixture
def job_manager():
    manager = MagicMock()
    manager.submit = AsyncMock(return_value="job-1")
    return manager


@pytest.fixture
def app(mock_registry, job_manager, tmp_path):
    application = create_app()
    application.state.registry = mock_registry
    application.state.build_store = BuildStore(tmp_path / "builds.json")
    application.state.job_manager = job_manager
    return application


@pytest.fixture
def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def make_model(name="Base", model_type=ModelType.CHECKPOINT):
    return ModelInfo(
        source=ModelSource.CIVITAI,
        id="123",
        name=name,
        base_architecture=BaseArchitecture.SD15,
        model_type=model_type,
        tags=[],
        download_url="",
        metadata={},
    )


def make_record(tmp_path, status=BuildStatus.COMPLETED, name="my-model"):
    recipe = Recipe(
        name=name,
        base_model=make_model(),
        loras=[],
        conversion_config=ConversionConfig(
            output_dir=tmp_path, model_name=name
        ),
    )
    return BuildRecord(recipe=recipe, status=status)


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_reports_ok_and_version(self, client):
        async with client as c:
            resp = await c.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"]
        assert isinstance(body["ml_deps_ok"], bool)
        assert isinstance(body["missing_deps"], list)


class TestUpload:
    @pytest.mark.asyncio
    async def test_register_path_does_not_copy(self, client, tmp_path):
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"x" * 2048)

        async with client as c:
            resp = await c.post("/api/upload", json={"path": str(checkpoint)})

        assert resp.status_code == 200
        body = resp.json()
        assert body["model_ref"].startswith("local_")
        assert body["name"] == "model"
        assert body["size_bytes"] == 2048

        registered = uploads.get(body["model_ref"])
        assert registered is not None
        # The file stays where it was — nothing was copied.
        assert registered.metadata["local_path"] == str(checkpoint.resolve())
        assert checkpoint.exists()

    @pytest.mark.asyncio
    async def test_register_path_rejects_bad_extension(self, client, tmp_path):
        bad = tmp_path / "notes.txt"
        bad.write_text("nope")

        async with client as c:
            resp = await c.post("/api/upload", json={"path": str(bad)})

        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "unsupported_format"

    @pytest.mark.asyncio
    async def test_register_path_rejects_missing_file(self, client, tmp_path):
        async with client as c:
            resp = await c.post(
                "/api/upload", json={"path": str(tmp_path / "ghost.safetensors")}
            )

        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_multipart_upload_copies_into_cache(
        self, client, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("COREML_CONVERTER_HOME", str(tmp_path / "app"))

        async with client as c:
            resp = await c.post(
                "/api/upload",
                files={"file": ("uploaded.safetensors", b"y" * 512)},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "uploaded"
        assert body["size_bytes"] == 512

        stored = uploads.get(body["model_ref"])
        assert stored.metadata["uploaded"] is True
        copied = Path(stored.metadata["local_path"])
        assert copied.is_file()
        assert copied.read_bytes() == b"y" * 512
        assert copied.parent == tmp_path / "app" / "cache" / "uploads"

    @pytest.mark.asyncio
    async def test_multipart_upload_without_file_is_rejected(self, client):
        async with client as c:
            resp = await c.post("/api/upload", data={"model_type": "checkpoint"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "no_file"


class TestBuildStart:
    @pytest.mark.asyncio
    async def test_starts_build_from_local_ref(
        self, client, app, tmp_path, job_manager
    ):
        checkpoint = tmp_path / "base.safetensors"
        checkpoint.write_bytes(b"z" * 64)
        model = uploads.register_path(checkpoint)

        async with client as c:
            resp = await c.post(
                "/api/build/start",
                json={
                    "name": "my-model",
                    "base": {"ref": model.id},
                    "output_dir": str(tmp_path / "out"),
                },
            )

        assert resp.status_code == 200
        build_id = resp.json()["build_id"]
        assert build_id

        job_manager.submit.assert_awaited_once()
        record = app.state.build_store.get(build_id)
        assert record is not None
        assert record.recipe.name == "my-model"
        assert record.recipe.conversion_config.output_dir == tmp_path / "out"
        # Defaults from the plan: split_einsum attention, all compute units.
        assert record.recipe.conversion_config.attention == "split_einsum"
        assert record.recipe.conversion_config.compute_units == "all"

    @pytest.mark.asyncio
    async def test_resolves_source_and_id(self, client, mock_registry, tmp_path):
        mock_registry.get_by_id.return_value = make_model(name="Remote")

        async with client as c:
            resp = await c.post(
                "/api/build/start",
                json={
                    "name": "remote-build",
                    "base": {"source": "civitai", "id": "349458"},
                    "output_dir": str(tmp_path),
                },
            )

        assert resp.status_code == 200
        mock_registry.get_by_id.assert_called_once()
        assert mock_registry.get_by_id.call_args[0][1] == "349458"

    @pytest.mark.asyncio
    async def test_includes_loras_with_weights(self, client, app, tmp_path):
        base = uploads.register_path(_write(tmp_path / "base.safetensors"))
        lora = uploads.register_path(
            _write(tmp_path / "style.safetensors"), model_type="lora"
        )

        async with client as c:
            resp = await c.post(
                "/api/build/start",
                json={
                    "name": "with-lora",
                    "base": {"ref": base.id},
                    "loras": [{"ref": lora.id, "weight": 0.7}],
                    "output_dir": str(tmp_path),
                },
            )

        assert resp.status_code == 200
        record = app.state.build_store.get(resp.json()["build_id"])
        assert len(record.recipe.loras) == 1
        assert record.recipe.loras[0].weight == 0.7

    @pytest.mark.asyncio
    async def test_unknown_ref_is_404(self, client, tmp_path):
        async with client as c:
            resp = await c.post(
                "/api/build/start",
                json={
                    "name": "x",
                    "base": {"ref": "local_nope"},
                    "output_dir": str(tmp_path),
                },
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "model_not_found"

    @pytest.mark.asyncio
    async def test_base_without_ref_or_source_is_400(self, client, tmp_path):
        async with client as c:
            resp = await c.post(
                "/api/build/start",
                json={"name": "x", "base": {}, "output_dir": str(tmp_path)},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_model_ref"

    @pytest.mark.asyncio
    async def test_missing_required_fields_is_400(self, client):
        async with client as c:
            resp = await c.post("/api/build/start", json={"name": "x"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_body"

    @pytest.mark.asyncio
    async def test_lora_weight_out_of_range_is_rejected(self, client, tmp_path):
        base = uploads.register_path(_write(tmp_path / "base.safetensors"))
        lora = uploads.register_path(
            _write(tmp_path / "style.safetensors"), model_type="lora"
        )

        async with client as c:
            resp = await c.post(
                "/api/build/start",
                json={
                    "name": "bad-weight",
                    "base": {"ref": base.id},
                    "loras": [{"ref": lora.id, "weight": 5.0}],
                    "output_dir": str(tmp_path),
                },
            )
        assert resp.status_code == 400


class TestBuildStatus:
    @pytest.mark.asyncio
    async def test_unknown_build_is_404(self, client):
        async with client as c:
            resp = await c.get("/api/build/does-not-exist/status")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "build_not_found"

    @pytest.mark.asyncio
    async def test_completed_build_reports_terminal_state(
        self, client, app, tmp_path
    ):
        record = make_record(tmp_path, status=BuildStatus.COMPLETED)
        app.state.build_store.save(record)

        async with client as c:
            resp = await c.get(f"/api/build/{record.id}/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["percent"] == 100
        assert body["error"] is None

    @pytest.mark.asyncio
    async def test_failed_build_surfaces_error(self, client, app, tmp_path):
        record = make_record(tmp_path, status=BuildStatus.FAILED)
        record.error = "converter exited with code 1"
        app.state.build_store.save(record)

        async with client as c:
            resp = await c.get(f"/api/build/{record.id}/status")

        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] == "converter exited with code 1"
        assert body["message"] == "converter exited with code 1"

    @pytest.mark.asyncio
    async def test_live_progress_is_reported(self, client, app, tmp_path):
        from coreml_converter.web.jobs import _set_progress

        record = make_record(tmp_path, status=BuildStatus.RUNNING)
        app.state.build_store.save(record)
        _set_progress(record.id, "converting", "Converting UNet", 45)

        async with client as c:
            resp = await c.get(f"/api/build/{record.id}/status")

        body = resp.json()
        assert body["status"] == "running"
        assert body["step"] == "converting"
        assert body["percent"] == 45


class TestBuildsList:
    @pytest.mark.asyncio
    async def test_empty_history(self, client):
        async with client as c:
            resp = await c.get("/api/builds")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_lists_builds_newest_first(self, client, app, tmp_path):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        older = make_record(tmp_path, name="older")
        older.started_at = now - timedelta(hours=2)
        newer = make_record(tmp_path, name="newer")
        newer.started_at = now

        app.state.build_store.save(older)
        app.state.build_store.save(newer)

        async with client as c:
            resp = await c.get("/api/builds")

        names = [b["name"] for b in resp.json()]
        assert names == ["newer", "older"]


def _write(path):
    path.write_bytes(b"w" * 32)
    return path


class TestDisplayName:
    """Fanny stores library files under an id-prefixed filename to keep
    same-named imports apart. Without an explicit name that prefix ends up in
    the build manifest the user reads."""

    @pytest.mark.asyncio
    async def test_explicit_name_overrides_the_filename(self, client, tmp_path):
        stored = tmp_path / "98F75B0B-uberRealisticPornMerge_v23Final.safetensors"
        stored.write_bytes(b"x" * 16)

        async with client as c:
            resp = await c.post(
                "/api/upload",
                json={"path": str(stored), "name": "uberRealisticPornMerge_v23Final"},
            )

        assert resp.status_code == 200
        assert resp.json()["name"] == "uberRealisticPornMerge_v23Final"
        assert uploads.get(resp.json()["model_ref"]).name == "uberRealisticPornMerge_v23Final"

    @pytest.mark.asyncio
    async def test_filename_is_used_when_no_name_is_given(self, client, tmp_path):
        stored = tmp_path / "plain.safetensors"
        stored.write_bytes(b"x" * 16)

        async with client as c:
            resp = await c.post("/api/upload", json={"path": str(stored)})

        assert resp.json()["name"] == "plain"


class TestDeleteBuilds:
    @pytest.mark.asyncio
    async def test_delete_one_build(self, client, app, tmp_path):
        record = make_record(tmp_path, status=BuildStatus.COMPLETED)
        app.state.build_store.save(record)

        async with client as c:
            resp = await c.delete(f"/api/builds/{record.id}")

        assert resp.status_code == 200
        assert app.state.build_store.get(record.id) is None

    @pytest.mark.asyncio
    async def test_delete_unknown_build_is_404(self, client):
        async with client as c:
            resp = await c.delete("/api/builds/nope")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_running_build_cannot_be_deleted(self, client, app, tmp_path):
        # Dropping it would orphan the job and leave the UI polling a dead id.
        record = make_record(tmp_path, status=BuildStatus.RUNNING)
        app.state.build_store.save(record)

        async with client as c:
            resp = await c.delete(f"/api/builds/{record.id}")

        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "build_running"
        assert app.state.build_store.get(record.id) is not None

    @pytest.mark.asyncio
    async def test_clear_finished_keeps_running_builds(self, client, app, tmp_path):
        done = make_record(tmp_path, status=BuildStatus.COMPLETED, name="done")
        failed = make_record(tmp_path, status=BuildStatus.FAILED, name="failed")
        running = make_record(tmp_path, status=BuildStatus.RUNNING, name="running")
        for r in (done, failed, running):
            app.state.build_store.save(r)

        async with client as c:
            resp = await c.delete("/api/builds")

        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2
        remaining = [r.id for r in app.state.build_store.list_all()]
        assert remaining == [running.id]
