from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from crackxnet_app.api import app
from crackxnet_app.deeppcb.model import create_faster_rcnn, load_checkpoint, save_checkpoint


def test_model_creation_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = create_faster_rcnn(pretrained=False, image_size=128)
    assert model.roi_heads.box_predictor.cls_score.out_features == 7
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(checkpoint, model, epoch=1, metrics={"f1": 0.0})
    loaded, payload = load_checkpoint(checkpoint, "cpu", pretrained=False)
    assert payload["epoch"] == 1
    assert loaded.roi_heads.box_predictor.cls_score.out_features == 7


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
