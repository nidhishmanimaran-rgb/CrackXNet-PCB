from __future__ import annotations

import pytest
import torch

from crackxnet_app.config import DDAFFConfig
from crackxnet_app.features.ddaff import DynamicDefectAwareFeatureFusion


def test_ddaff_construction_and_output_shape() -> None:
    module = DynamicDefectAwareFeatureFusion(
        local_channels=1280,
        global_dim=768,
        config=DDAFFConfig(fusion_dim=64, dropout=0.0, use_layer_norm=True, device="cpu"),
    )
    local = torch.rand(2, 1280, 4, 4)
    global_tokens = torch.rand(2, 16, 768)

    output = module(local, global_tokens)

    assert output.fused_map.shape == (2, 64, 4, 4)
    assert output.pooled_features.shape == (2, 64)
    assert output.local_projected.shape == (2, 64, 4, 4)
    assert output.global_projected.shape == (2, 64, 4, 4)
    assert output.fusion_weights.shape == (2, 2)
    assert torch.allclose(output.fusion_weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert output.fused_map.numel() > 0
    assert torch.isfinite(output.fused_map).all()


def test_ddaff_supports_mismatched_dimensions_and_batch() -> None:
    module = DynamicDefectAwareFeatureFusion(
        local_channels=32,
        global_dim=11,
        fusion_dim=17,
        dropout=0.0,
        device="cpu",
    )
    output = module(torch.rand(3, 32, 5, 7), torch.rand(3, 9, 11))

    assert output.fused_map.shape == (3, 17, 5, 7)
    assert output.global_projected.shape == (3, 17, 5, 7)


def test_ddaff_has_trainable_fusion_parameters_and_gradients() -> None:
    module = DynamicDefectAwareFeatureFusion(local_channels=8, global_dim=6, fusion_dim=5, dropout=0.0, device="cpu")
    local = torch.rand(2, 8, 3, 3, requires_grad=True)
    global_tokens = torch.rand(2, 9, 6, requires_grad=True)

    output = module(local, global_tokens)
    loss = output.fused_map.square().mean() + output.fusion_weights.mean()
    loss.backward()

    assert module.stream_logits.requires_grad
    assert module.stream_logits.grad is not None
    assert torch.isfinite(module.stream_logits.grad).all()
    assert local.grad is not None
    assert global_tokens.grad is not None
    assert module.local_projection.weight.grad is not None
    assert module.global_projection.weight.grad is not None


def test_ddaff_eval_is_deterministic() -> None:
    module = DynamicDefectAwareFeatureFusion(local_channels=4, global_dim=4, fusion_dim=4, dropout=0.0, device="cpu")
    module.eval()
    local = torch.rand(1, 4, 2, 2)
    global_tokens = torch.rand(1, 4, 4)

    with torch.no_grad():
        first = module(local, global_tokens)
        second = module(local, global_tokens)

    assert torch.allclose(first.fused_map, second.fused_map)
    assert torch.allclose(first.fusion_weights, second.fusion_weights)


def test_ddaff_invalid_shape_handling() -> None:
    module = DynamicDefectAwareFeatureFusion(local_channels=4, global_dim=4, fusion_dim=4, device="cpu")

    with pytest.raises(ValueError, match="NCHW"):
        module(torch.rand(4, 2, 2), torch.rand(1, 4, 4))
    with pytest.raises(ValueError, match="BNC"):
        module(torch.rand(1, 4, 2, 2), torch.rand(1, 4, 2, 2))
    with pytest.raises(ValueError, match="square"):
        module(torch.rand(1, 4, 2, 2), torch.rand(1, 5, 4))
    with pytest.raises(ValueError, match="matching batch"):
        module(torch.rand(2, 4, 2, 2), torch.rand(1, 4, 4))


def test_ddaff_config_loading() -> None:
    config = DDAFFConfig(fusion_dim=31, dropout=0.1, use_layer_norm=False, device="cpu")
    module = DynamicDefectAwareFeatureFusion(local_channels=8, global_dim=8, config=config)

    assert module.config_snapshot.fusion_dim == 31
    assert module.config_snapshot.dropout == 0.1
    assert module.config_snapshot.use_layer_norm is False
    assert module.config_snapshot.device == "cpu"
