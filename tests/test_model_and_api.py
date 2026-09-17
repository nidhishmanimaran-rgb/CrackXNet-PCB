from __future__ import annotations

from pathlib import Path
import random

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import torch

from crackxnet_app.api import app
from crackxnet_app import api
from crackxnet_app.config import SecurityConfig
from crackxnet_app.deeppcb import model as deeppcb_model
from crackxnet_app.deeppcb.model import create_faster_rcnn, load_checkpoint, save_checkpoint, set_reproducible_seed
from crackxnet_app.inference.faster_rcnn_detector import FasterRCNNInspectionDetector
from crackxnet_app.inference import preprocessing
from crackxnet_app.inference.pipeline import CrackXNetPipeline, DetectorUnavailableError


def test_model_creation_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = create_faster_rcnn(pretrained=False, image_size=128)
    assert model.roi_heads.box_predictor.cls_score.out_features == 7
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(checkpoint, model, epoch=1, metrics={"f1": 0.0}, training_config={"seed": 42})
    loaded, payload = load_checkpoint(checkpoint, "cpu", pretrained=False)
    assert payload["epoch"] == 1
    assert payload["training_config"]["seed"] == 42
    assert loaded.roi_heads.box_predictor.cls_score.out_features == 7


def test_baseline_detector_uses_checkpoint_image_size(tmp_path: Path) -> None:
    model = create_faster_rcnn(pretrained=False, image_size=128)
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(checkpoint, model, image_size=128)

    detector = FasterRCNNInspectionDetector(checkpoint, device="cpu")

    assert detector.image_size == 128


def test_api_health_and_demo_inspect(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (96, 96), (30, 100, 60)).save(image_path)
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    with image_path.open("rb") as handle:
        response = client.post("/api/inspect", files={"file": ("sample.png", handle, "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] in {"PASS", "REWORK", "REJECT"}
    assert "model_status" in payload


def test_api_sanitizes_uploaded_display_filename(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), (30, 100, 60)).save(image_path)
    client = TestClient(app)
    with image_path.open("rb") as handle:
        response = client.post("/api/inspect", files={"file": ("..\\secret\\sample.png", handle, "image/png")})

    assert response.status_code == 200
    assert response.json()["filename"] == "sample.png"


def test_checkpoint_loader_rejects_non_checkpoint_suffix(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match=".pt or .pth"):
        deeppcb_model.load_checkpoint_payload(path)


def test_checkpoint_loader_rejects_wrong_class_count(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong_classes.pth"
    torch.save({"model_state": {}, "num_classes": 99}, checkpoint)

    with pytest.raises(RuntimeError, match="class count"):
        load_checkpoint(checkpoint, "cpu", pretrained=False)


def test_upload_pixel_limit_and_report_cache_bound(monkeypatch) -> None:
    image = Image.new("RGB", (4, 4), (1, 2, 3))
    import io

    payload = io.BytesIO()
    image.save(payload, format="PNG")
    monkeypatch.setattr(preprocessing, "MAX_UPLOAD_PIXELS", 10)
    with pytest.raises(ValueError, match="pixel limit"):
        preprocessing.load_rgb_image(payload.getvalue())

    from crackxnet_app import api

    api._REPORT_CACHE.clear()
    monkeypatch.setattr(api, "REPORT_CACHE_MAX_ENTRIES", 2)
    api._cache_report("one", "1")
    api._cache_report("two", "2")
    api._cache_report("three", "3")
    assert list(api._REPORT_CACHE) == ["two", "three"]


def test_reproducible_seed_sets_python_numpy_and_torch() -> None:
    set_reproducible_seed(7)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_reproducible_seed(7)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_runtime_environment_is_weights_only_serializable(tmp_path: Path) -> None:
    model = create_faster_rcnn(pretrained=False, image_size=64)
    checkpoint = tmp_path / "runtime.pth"
    save_checkpoint(checkpoint, model, training_config={"runtime_environment": deeppcb_model.runtime_environment()})

    _, payload = load_checkpoint(checkpoint, "cpu", pretrained=False)

    assert isinstance(payload["training_config"]["runtime_environment"]["torch"], str)


def test_invalid_existing_checkpoint_does_not_silently_fall_back(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.pth"
    checkpoint.write_bytes(b"not a checkpoint")
    pipeline = CrackXNetPipeline(checkpoint_path=checkpoint, model_mode="baseline")

    assert pipeline.model_status == "Configured detector unavailable"
    with pytest.raises(DetectorUnavailableError):
        pipeline.inspect(Image.new("RGB", (16, 16)))


def test_optional_api_key_and_rate_limit(monkeypatch) -> None:
    api._REQUEST_TIMESTAMPS.clear()
    monkeypatch.setattr(api, "DEFAULT_SECURITY_CONFIG", SecurityConfig(api_key="test-key", rate_limit_requests=2, rate_limit_window_seconds=60))
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    unauthorized = client.get("/api/report/not-found")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/report/not-found", headers={"X-API-Key": "test-key"}).status_code == 404
    assert client.get("/api/report/not-found", headers={"X-API-Key": "test-key"}).status_code == 404
    limited = client.get("/api/report/not-found", headers={"X-API-Key": "test-key"})
    assert limited.status_code == 429
    assert limited.headers["x-frame-options"] == "DENY"


def test_security_headers_are_set() -> None:
    response = TestClient(app).get("/")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
