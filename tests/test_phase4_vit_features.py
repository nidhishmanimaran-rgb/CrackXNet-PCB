from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from crackxnet_app.config import GlobalFeatureConfig
from crackxnet_app.features import vit
from crackxnet_app.features.vit import ViTFeatureBackbone, ViTGlobalFeatureExtractor, ViTVariantSpec
from crackxnet_app.inference.modules import TorchGlobalFeatureExtractor


def test_vit_pretrained_false_passes_no_weights(monkeypatch) -> None:
    captured = {}

    class DummyViT(nn.Module):
        hidden_dim = 8

        def __init__(self) -> None:
            super().__init__()
            self.class_token = nn.Parameter(torch.zeros(1, 1, 8))
            self.encoder = nn.Identity()

        def _process_input(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 4, 8)

    @dataclass(frozen=True)
    class DummyWeights:
        DEFAULT: object = object()

    def fake_builder(*, weights, image_size):
        captured["weights"] = weights
        captured["image_size"] = image_size
        return DummyViT()

    monkeypatch.setitem(
        vit.VIT_VARIANTS,
        "dummy_vit",
        ViTVariantSpec(builder=fake_builder, weights_enum=DummyWeights, patch_size=16),
    )

    backbone = ViTFeatureBackbone("dummy_vit", pretrained=False, image_size=64)

    assert captured["weights"] is None
    assert captured["image_size"] == 64
    assert backbone.hidden_dim == 8


def test_vit_global_feature_forward_cpu_batch() -> None:
    extractor = ViTGlobalFeatureExtractor(
        config=GlobalFeatureConfig(
            vit_variant="vit_b_16",
            vit_pretrained=False,
            input_size=64,
            patch_size=16,
            device="cpu",
            checkpoint_path=None,
        )
    )
    image = torch.rand(2, 3, 64, 64)

    output = extractor.extract_global_features(image)

    assert output.backbone_name == "vit_b_16"
    assert output.input_shape == (2, 3, 64, 64)
    assert output.patch_size == 16
    assert output.class_token.shape == (2, 768)
    assert output.patch_tokens.shape == (2, 16, 768)
    assert output.pooled_features.shape == (2, 768)
    assert output.patch_tokens.numel() > 0
    assert torch.isfinite(output.class_token).all()
    assert torch.isfinite(output.patch_tokens).all()
    assert output.class_token.device.type == "cpu"


def test_vit_rejects_wrong_input_size() -> None:
    extractor = ViTGlobalFeatureExtractor(pretrained=False, input_size=64, device="cpu")

    try:
        extractor.extract_global_features(torch.rand(1, 3, 32, 32))
    except ValueError as exc:
        assert "64x64" in str(exc)
    else:
        raise AssertionError("ViT should reject tensors that do not match configured input_size")


def test_vit_config_and_protocol_compatibility() -> None:
    config = GlobalFeatureConfig(
        vit_variant="vit_b_32",
        vit_pretrained=False,
        input_size=64,
        patch_size=32,
        device="cpu",
        checkpoint_path=None,
    )
    extractor = ViTGlobalFeatureExtractor(config=config)

    assert config.vit_variant == "vit_b_32"
    assert config.input_size == 64
    assert isinstance(extractor, TorchGlobalFeatureExtractor)
