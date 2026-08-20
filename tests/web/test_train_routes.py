"""JSON API for training runs."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from coreml_converter.core.models import TrainRecord, TrainRequest, TrainStatus
from coreml_converter.core.state import TrainStore
from coreml_converter.web.app import create_app


@pytest.fixture
def images(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(12):
        Image.new("RGB", (600, 900), (100 + i, 90, 80)).save(folder / f"{i:02d}.png")
    return folder


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "base.safetensors"
    path.write_bytes(b"\x00" * 16)
    return path


@pytest.fixture
def store(tmp_path):
    return TrainStore(tmp_path / "trainings.json")


@pytest.fixture
def manager(store):
    mock = MagicMock()
    mock.store = store
    mock.active_count.return_value = 0

    async def submit(record):
        record.status = TrainStatus.RUNNING
        store.save(record)
        return record.id

    mock.submit = AsyncMock(side_effect=submit)
    return mock


@pytest.fixture
def client(store, manager):
    app = create_app()
    app.state.registry = MagicMock()
    app.state.train_store = store
    app.state.train_manager = manager
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def body(images, checkpoint, tmp_path, **overrides):
    payload = {
        "name": "leah", "trigger": "fnnyleah", "class_token": "woman",
        "image_paths": [str(images)], "base_path": str(checkpoint),
        "output_dir": str(tmp_path / "out"),
    }
    payload.update(overrides)
    return payload


async def post(client, payload):
    async with client as c:
        return await c.post("/api/train/start", json=payload)


@pytest.mark.asyncio
async def test_start_accepts_a_valid_request(client, images, checkpoint, tmp_path):
    response = await post(client, body(images, checkpoint, tmp_path))
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "character"
    assert data["images_found"] == 12
    assert data["train_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value,code", [
    ("name", "", "name_required"),
    ("trigger", "", "trigger_required"),
    ("mode", "banana", "bad_mode"),
    ("base_path", "/does/not/exist", "base_not_found"),
])
async def test_start_rejects_bad_input(client, images, checkpoint, tmp_path,
                                       field, value, code):
    response = await post(client, body(images, checkpoint, tmp_path, **{field: value}))
    assert response.status_code in (400, 412)
    assert response.json()["error"]["code"] == code


@pytest.mark.asyncio
async def test_too_few_images_returns_412(client, checkpoint, tmp_path):
    """Matches the design spec's precondition for too few references."""
    thin = tmp_path / "thin"
    thin.mkdir()
    for i in range(3):
        Image.new("RGB", (600, 900)).save(thin / f"{i}.png")
    response = await post(client, body(thin, checkpoint, tmp_path))
    assert response.status_code == 412
    assert response.json()["error"]["code"] == "not_enough_images"


@pytest.mark.asyncio
async def test_unknown_params_are_rejected(client, images, checkpoint, tmp_path):
    response = await post(client, body(images, checkpoint, tmp_path,
                                       params={"nonsense": 1}))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_params"


@pytest.mark.asyncio
async def test_mode_selects_its_preset_step_count(client, images, checkpoint, tmp_path):
    response = await post(client, body(images, checkpoint, tmp_path, mode="pose"))
    assert response.json()["mode"] == "pose"
    assert response.json()["steps"] == 1200


@pytest.mark.asyncio
async def test_second_run_is_refused_while_one_is_active(
        client, manager, images, checkpoint, tmp_path):
    manager.active_count.return_value = 1
    response = await post(client, body(images, checkpoint, tmp_path))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "train_busy"


@pytest.mark.asyncio
async def test_status_reports_a_known_run(client, store, tmp_path):
    record = TrainRecord(request=TrainRequest(
        name="n", trigger="t", image_paths=["/a.png"], output_dir=tmp_path))
    store.save(record)
    async with client as c:
        response = await c.get(f"/api/train/{record.id}/status")
    assert response.status_code == 200
    assert response.json()["train_id"] == record.id


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", [
    ("get", "/api/train/missing/status"),
    ("post", "/api/train/missing/cancel"),
    ("delete", "/api/trains/missing"),
])
async def test_unknown_id_is_404(client, method, path):
    async with client as c:
        response = await getattr(c, method)(path)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "train_not_found"


@pytest.mark.asyncio
async def test_cancel_refuses_a_finished_run(client, store, tmp_path):
    record = TrainRecord(
        request=TrainRequest(name="n", trigger="t", image_paths=["/a.png"],
                             output_dir=tmp_path),
        status=TrainStatus.COMPLETED)
    store.save(record)
    async with client as c:
        response = await c.post(f"/api/train/{record.id}/cancel")
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_delete_refuses_a_running_run(client, store, tmp_path):
    record = TrainRecord(
        request=TrainRequest(name="n", trigger="t", image_paths=["/a.png"],
                             output_dir=tmp_path),
        status=TrainStatus.RUNNING)
    store.save(record)
    async with client as c:
        response = await c.delete(f"/api/trains/{record.id}")
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_history_is_newest_first(client, store, tmp_path):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    for i, delta in enumerate([timedelta(hours=2), timedelta(hours=1)]):
        record = TrainRecord(
            request=TrainRequest(name=f"n{i}", trigger="t",
                                 image_paths=["/a.png"], output_dir=tmp_path),
            status=TrainStatus.COMPLETED, started_at=now - delta)
        store.save(record)
    async with client as c:
        response = await c.get("/api/trains")
    names = [r["name"] for r in response.json()]
    assert names == ["n1", "n0"]


# --- dataset inspection ---------------------------------------------------

@pytest.mark.asyncio
async def test_inspect_reports_counts(client, images):
    async with client as c:
        response = await c.post("/api/dataset/inspect",
                                json={"image_paths": [str(images)]})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 12
    assert len(body["images"]) == 12
    assert all("name" in i and "flagged" in i for i in body["images"])


@pytest.mark.asyncio
async def test_inspect_requires_paths(client):
    async with client as c:
        response = await c.post("/api/dataset/inspect", json={})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "images_required"


@pytest.mark.asyncio
async def test_inspect_rejects_an_empty_folder(client, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    async with client as c:
        response = await c.post("/api/dataset/inspect",
                                json={"image_paths": [str(empty)]})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "dataset_unusable"


@pytest.mark.asyncio
async def test_inspect_does_not_train(client, images, manager):
    """Inspection must be free of side effects — it runs before the user commits."""
    async with client as c:
        await c.post("/api/dataset/inspect", json={"image_paths": [str(images)]})
    manager.submit.assert_not_called()
