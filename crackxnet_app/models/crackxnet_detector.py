from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

from crackxnet_app.config import (
    DDAFFConfig,
    GlobalFeatureConfig,
    HybridDetectorConfig,
    LocalFeatureConfig,
    NUM_DETECTION_CLASSES,
)
from crackxnet_app.deeppcb.model import get_device
from crackxnet_app.features.ddaff import DynamicDefectAwareFeatureFusion
from crackxnet_app.features.efficientnet_cbam import EfficientNetCBAMLocalFeatureExtractor
from crackxnet_app.features.vit import ViTGlobalFeatureExtractor, VIT_VARIANTS


class CrackXNetHybridBackbone(nn.Module):
    """EfficientNet-CBAM + ViT + DDAFF + FPN backbone for Faster R-CNN.

    Input: `(batch, 3, image_size, image_size)`.
    DDAFF output: `(batch, fusion_dim, h, w)`.
    FPN output: `OrderedDict({"0": (batch, fpn_out_channels, h, w)})`.
    """

    def __init__(self, config: HybridDetectorConfig) -> None:
        super().__init__()
        self.config = config
        if config.image_size != config.local_input_size or config.image_size != config.vit_input_size:
            raise ValueError("Hybrid detector currently requires image_size, local_input_size, and vit_input_size to match.")
        if config.vit_variant not in VIT_VARIANTS:
            raise ValueError(f"Unsupported ViT variant: {config.vit_variant}")
        patch_size = VIT_VARIANTS[config.vit_variant].patch_size
        self.local_extractor = EfficientNetCBAMLocalFeatureExtractor(
            config=LocalFeatureConfig(
                efficientnet_pretrained=config.local_pretrained,
                input_size=config.local_input_size,
                cbam_reduction_ratio=16,
                device=config.device,
                checkpoint_path=None,
            )
        )
        self.global_extractor = ViTGlobalFeatureExtractor(
            config=GlobalFeatureConfig(
                vit_variant=config.vit_variant,
                vit_pretrained=config.vit_pretrained,
                input_size=config.vit_input_size,
                patch_size=patch_size,
                device=config.device,
                checkpoint_path=None,
            )
        )
        self.fusion = DynamicDefectAwareFeatureFusion(
            local_channels=1280,
            global_dim=768,
            config=DDAFFConfig(fusion_dim=config.fusion_dim, dropout=0.0, use_layer_norm=True, device=config.device),
        )
        self.fpn = FeaturePyramidNetwork([config.fusion_dim], config.fpn_out_channels)
        self.out_channels = config.fpn_out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"CrackXNetHybridBackbone expects BCHW RGB tensor, got {tuple(x.shape)}.")
        if x.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(
                f"CrackXNetHybridBackbone expected {self.config.image_size}x{self.config.image_size}, "
                f"got {tuple(x.shape[-2:])}."
            )
        local = self.local_extractor(x)
        global_features = self.global_extractor(x)
        fused = self.fusion(local.feature_map, global_features.patch_tokens)
        return self.fpn(OrderedDict([("0", fused.fused_map)]))

    def feature_diagnostics(self, x: torch.Tensor) -> dict[str, object]:
        with torch.no_grad():
            local = self.local_extractor(x)
            global_features = self.global_extractor(x)
            fused = self.fusion(local.feature_map, global_features.patch_tokens)
            fpn = self.fpn(OrderedDict([("0", fused.fused_map)]))
        return {
            "input_shape": tuple(x.shape),
            "local_shape": tuple(local.feature_map.shape),
            "global_tokens_shape": tuple(global_features.patch_tokens.shape),
            "fused_shape": tuple(fused.fused_map.shape),
            "fpn_shapes": {name: tuple(value.shape) for name, value in fpn.items()},
            "fusion_weights": fused.fusion_weights.detach().cpu().tolist(),
        }


def create_hybrid_faster_rcnn(
    config: HybridDetectorConfig,
    num_classes: int = NUM_DETECTION_CLASSES,
) -> FasterRCNN:
    backbone = CrackXNetHybridBackbone(config)
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=config.image_size,
        max_size=config.image_size,
    )


def hybrid_config_to_dict(config: HybridDetectorConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "checkpoint_path": str(config.checkpoint_path),
        "image_size": config.image_size,
        "fusion_dim": config.fusion_dim,
        "fpn_out_channels": config.fpn_out_channels,
        "local_input_size": config.local_input_size,
        "vit_input_size": config.vit_input_size,
        "vit_variant": config.vit_variant,
        "local_pretrained": config.local_pretrained,
        "vit_pretrained": config.vit_pretrained,
        "confidence_threshold": config.confidence_threshold,
        "device": config.device,
    }


def hybrid_config_from_dict(data: dict[str, Any], device: str | None = None) -> HybridDetectorConfig:
    return HybridDetectorConfig(
        enabled=bool(data.get("enabled", True)),
        checkpoint_path=Path(data.get("checkpoint_path", "outputs/hybrid_checkpoints/best.pth")),
        image_size=int(data.get("image_size", 224)),
        fusion_dim=int(data.get("fusion_dim", 256)),
        fpn_out_channels=int(data.get("fpn_out_channels", 256)),
        local_input_size=int(data.get("local_input_size", data.get("image_size", 224))),
        vit_input_size=int(data.get("vit_input_size", data.get("image_size", 224))),
        vit_variant=str(data.get("vit_variant", "vit_b_16")),
        local_pretrained=bool(data.get("local_pretrained", False)),
        vit_pretrained=bool(data.get("vit_pretrained", False)),
        confidence_threshold=float(data.get("confidence_threshold", 0.5)),
        device=device or str(data.get("device", "auto")),
    )


def save_hybrid_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: HybridDetectorConfig | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if config is None:
        backbone = getattr(model, "backbone", None)
        config = getattr(backbone, "config", None)
    if config is None:
        raise ValueError("Hybrid checkpoint save requires a HybridDetectorConfig.")
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "num_classes": NUM_DETECTION_CLASSES,
        "model_name": "crackxnet_hybrid_faster_rcnn",
        "model_mode": "hybrid",
        "hybrid_config": hybrid_config_to_dict(config),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_hybrid_checkpoint(path: str | Path, device: str = "auto") -> tuple[FasterRCNN, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hybrid checkpoint not found: {path}")
    torch_device = get_device(device)
    payload = torch.load(path, map_location=torch_device)
    if payload.get("model_mode") != "hybrid" or "hybrid_config" not in payload:
        raise RuntimeError(f"Invalid hybrid checkpoint: {path}")
    config = hybrid_config_from_dict(payload["hybrid_config"], device=str(torch_device))
    model = create_hybrid_faster_rcnn(config, num_classes=int(payload.get("num_classes", NUM_DETECTION_CLASSES)))
    model.load_state_dict(payload["model_state"])
    model.to(torch_device)
    model.eval()
    return model, payload
