from __future__ import annotations

from PIL import Image, ImageDraw

from crackxnet_app.inference.pipeline import CrackXNetPipeline, DetectorUnavailableError
import pytest


def test_pipeline_returns_reportable_result() -> None:
    image = Image.new("RGB", (220, 160), (30, 105, 65))
    draw = ImageDraw.Draw(image)
    draw.rectangle([35, 45, 185, 55], fill=(184, 112, 38))
    draw.rectangle([102, 42, 112, 60], fill=(245, 245, 235))
    draw.ellipse([165, 90, 176, 101], fill=(5, 5, 5))

    outputs = CrackXNetPipeline(force_demo=True).inspect(image, filename="synthetic.png")

    assert outputs.result.filename == "synthetic.png"
    assert outputs.result.decision in {"PASS", "REWORK", "REJECT"}
    assert outputs.overlay.size == image.size
    assert outputs.heatmap.size == image.size
    assert isinstance(outputs.result.defects, list)


def test_blank_image_can_pass_without_crashing() -> None:
    image = Image.new("RGB", (128, 128), (45, 115, 70))
    outputs = CrackXNetPipeline(force_demo=True).inspect(image)

    assert outputs.result.image_width == 128
    assert outputs.result.image_height == 128
    assert outputs.result.max_severity >= 0.0


def test_missing_default_checkpoint_does_not_fall_back_to_demo(tmp_path) -> None:
    pipeline = CrackXNetPipeline(checkpoint_path=tmp_path / "missing.pth", model_mode="hybrid")

    assert pipeline.initialization_error is True
    assert pipeline.status_dict()["checkpoint_status"] == "missing"
    assert "Model unavailable" in pipeline.model_status
    with pytest.raises(DetectorUnavailableError):
        pipeline.inspect(Image.new("RGB", (64, 64), (45, 115, 70)))
