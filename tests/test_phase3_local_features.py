from __future__ import annotations

import torch
from torch import nn

from crackxnet_app.config import LocalFeatureConfig
from crackxnet_app.features.cbam import CBAM
from crackxnet_app.features import efficientnet_cbam
from crackxnet_app.features.efficientnet_cbam import EfficientNetB0FeatureBackbone, EfficientNetCBAMLocalFeatureExtractor


def test_cbam_preserves_tensor_shape() -> None:
    module = CBAM(channels=16, reduction_ratio=4)
    x = torch.rand(2, 16, 12, 10)

    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_cbam_rejects_non_nchw_tensor() -> None:
    module = CBAM(channels=8)
    try:
        module(torch.rand(8, 12, 12))
    except ValueError as exc:
        assert "NCHW" in str(exc)
    else:
        raise AssertionError("CBAM should reject non-NCHW tensors")


def test_efficientnet_pretrained_false_passes_no_weights(monkeypatch) -> None:
    captured = {}

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Identity()

    def fake_efficientnet_b0(*, weights):
        captured["weights"] = weights
        return DummyModel()

    monkeypatch.setattr(efficientnet_cbam, "efficientnet_b0", fake_efficientnet_b0)

    backbone = EfficientNetB0FeatureBackbone(pretrained=False)

    assert captured["weights"] is None
    assert isinstance(backbone.features, nn.Identity)


def test_combined_efficientnet_cbam_forward_cpu() -> None:
    extractor = EfficientNetCBAMLocalFeatureExtractor(
        config=LocalFeatureConfig(
            efficientnet_pretrained=False,
            input_size=96,
            cbam_reduction_ratio=16,
            device="cpu",
            checkpoint_path=None,
        )
    )
    image = torch.rand(1, 3, 96, 96)

    output = extractor.extract_local_features(image)

    assert output.backbone_name == "efficientnet_b0_cbam"
    assert output.input_shape == (1, 3, 96, 96)
    assert output.feature_map.ndim == 4
    assert output.feature_map.shape[0] == 1
    assert output.feature_map.shape[1] == 1280
    assert output.feature_map.numel() > 0
    assert output.pooled_features.shape == (1, 1280)
    assert output.feature_map.device.type == "cpu"


def test_local_feature_config_values() -> None:
    config = LocalFeatureConfig(
        efficientnet_pretrained=False,
        input_size=128,
        cbam_reduction_ratio=8,
        device="cpu",
        checkpoint_path=None,
    )

    assert config.input_size == 128
    assert config.cbam_reduction_ratio == 8
    assert config.device == "cpu"
