from __future__ import annotations

from PIL import Image, ImageDraw

from crackxnet_app.inference.pipeline import CrackXNetPipeline


def test_pipeline_returns_reportable_result() -> None:
    image = Image.new("RGB", (220, 160), (30, 105, 65))
    draw = ImageDraw.Draw(image)
    draw.rectangle([35, 45, 185, 55], fill=(184, 112, 38))
    draw.rectangle([102, 42, 112, 60], fill=(245, 245, 235))
    draw.ellipse([165, 90, 176, 101], fill=(5, 5, 5))

    outputs = CrackXNetPipeline().inspect(image, filename="synthetic.png")

    assert outputs.result.filename == "synthetic.png"
    assert outputs.result.decision in {"PASS", "REWORK", "REJECT"}
    assert outputs.overlay.size == image.size
    assert outputs.heatmap.size == image.size
    assert isinstance(outputs.result.defects, list)


def test_blank_image_can_pass_without_crashing() -> None:
    image = Image.new("RGB", (128, 128), (45, 115, 70))
    outputs = CrackXNetPipeline().inspect(image)

    assert outputs.result.image_width == 128
    assert outputs.result.image_height == 128
    assert outputs.result.max_severity >= 0.0
