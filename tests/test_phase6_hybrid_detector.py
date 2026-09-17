from __future__ import annotations

from pathlib import Path

import pytest
import torch

from crackxnet_app.config import HybridDetectorConfig
from crackxnet_app.models.crackxnet_detector import (
    CrackXNetHybridBackbone,
    create_hybrid_faster_rcnn,
    load_hybrid_checkpoint,
    save_hybrid_checkpoint,
)


def _config(tmp_path: Path | None = None) -> HybridDetectorConfig:
    return HybridDetectorConfig(
        enabled=True,
        checkpoint_path=(tmp_path / "hybrid.pth") if tmp_path else Path("outputs/hybrid_test/best.pth"),
        image_size=64,
        fusion_dim=16,
        fpn_out_channels=16,
        local_input_size=64,
        vit_input_size=64,
        vit_variant="vit_b_16",
        local_pretrained=False,
        vit_pretrained=False,
        confidence_threshold=0.0,
        device="cpu",
    )


def _targets(batch_size: int = 2) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor([[8.0, 8.0, 28.0, 30.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        for _ in range(batch_size)
    ]


def test_hybrid_backbone_fpn_shapes() -> None:
    backbone = CrackXNetHybridBackbone(_config())
    features = backbone(torch.rand(2, 3, 64, 64))

    assert list(features.keys()) == ["0"]
    assert features["0"].shape == (2, 16, 2, 2)
    assert torch.isfinite(features["0"]).all()
    assert backbone.out_channels == 16


def test_hybrid_backbone_diagnostics() -> None:
    backbone = CrackXNetHybridBackbone(_config())
    diagnostics = backbone.feature_diagnostics(torch.rand(1, 3, 64, 64))

    assert diagnostics["local_shape"] == (1, 1280, 2, 2)
    assert diagnostics["global_tokens_shape"] == (1, 16, 768)
    assert diagnostics["fused_shape"] == (1, 16, 2, 2)
    assert diagnostics["fpn_shapes"]["0"] == (1, 16, 2, 2)


def test_hybrid_detector_training_loss_and_backward() -> None:
    model = create_hybrid_faster_rcnn(_config())
    model.train()
    images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    losses = model(images, _targets(2))
    loss = sum(losses.values())
    loss.backward()

    assert {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}.issubset(losses.keys())
    assert torch.isfinite(loss.detach())
    assert model.backbone.fusion.stream_logits.grad is not None


def test_hybrid_detector_inference_output_format() -> None:
    model = create_hybrid_faster_rcnn(_config())
    model.eval()
    with torch.no_grad():
        outputs = model([torch.rand(3, 64, 64)])

    assert isinstance(outputs, list)
    assert set(outputs[0].keys()) == {"boxes", "labels", "scores"}
    assert outputs[0]["boxes"].ndim == 2
    assert outputs[0]["labels"].ndim == 1
    assert outputs[0]["scores"].ndim == 1


def test_hybrid_checkpoint_save_load_and_inference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = create_hybrid_faster_rcnn(config)
    checkpoint = tmp_path / "hybrid.pth"
    save_hybrid_checkpoint(checkpoint, model, epoch=1, metrics={"loss": 0.0}, config=config)

    loaded, payload = load_hybrid_checkpoint(checkpoint, "cpu")
    loaded.eval()
    with torch.no_grad():
        outputs = loaded([torch.rand(3, 64, 64)])

    assert payload["model_mode"] == "hybrid"
    assert payload["epoch"] == 1
    assert outputs[0]["boxes"].ndim == 2


def test_hybrid_checkpoint_rejects_wrong_class_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = create_hybrid_faster_rcnn(config)
    checkpoint = tmp_path / "hybrid.pth"
    save_hybrid_checkpoint(checkpoint, model, epoch=1, metrics={"loss": 0.0}, config=config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["num_classes"] = 99
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="class count"):
        load_hybrid_checkpoint(checkpoint, "cpu")


def test_hybrid_backbone_invalid_shape() -> None:
    backbone = CrackXNetHybridBackbone(_config())

    with pytest.raises(ValueError, match="BCHW"):
        backbone(torch.rand(3, 64, 64))
    with pytest.raises(ValueError, match="64x64"):
        backbone(torch.rand(1, 3, 32, 32))
