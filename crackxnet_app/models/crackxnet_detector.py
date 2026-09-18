from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0
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
from crackxnet_app.deeppcb.model import get_device, load_checkpoint_payload
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


class _ProductionChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_attention = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_attention = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg_attention + max_attention)


class _ProductionSpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_values, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, max_values], dim=1)))


class _ProductionCBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = _ProductionChannelAttention(channels)
        self.spatial = _ProductionSpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        return x * self.spatial(x)


class _ProductionDDAFFGate(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, local: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        if global_context.shape[-2:] != local.shape[-2:]:
            global_context = F.interpolate(
                global_context,
                size=local.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        weights = torch.softmax(self.weight_net(torch.cat([local, global_context], dim=1)), dim=1)
        return weights[:, :1] * local + weights[:, 1:] * global_context


class _ProductionViTContext(nn.Module):
    def __init__(self, channels: int = 256, feedforward_dim: int = 1024, layers: int = 2) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, channels, 20, 20))
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        position = self.position
        if position.shape[-2:] != x.shape[-2:]:
            position = F.interpolate(position, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x + position
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens)
        return encoded.transpose(1, 2).reshape(batch, channels, height, width)


class ProductionCrackXNetBackbone(nn.Module):
    """Backbone matching the verified 10-epoch production checkpoint schema."""

    out_channels = 256

    def __init__(self) -> None:
        super().__init__()
        self.features = efficientnet_b0(weights=None).features
        self.cbam = nn.ModuleList([_ProductionCBAM(40), _ProductionCBAM(112), _ProductionCBAM(192)])
        self.projection = nn.ModuleList(
            [
                nn.Conv2d(40, self.out_channels, kernel_size=1),
                nn.Conv2d(112, self.out_channels, kernel_size=1),
                nn.Conv2d(192, self.out_channels, kernel_size=1),
            ]
        )
        self.vit = _ProductionViTContext(self.out_channels)
        self.ddaff = nn.ModuleList([_ProductionDDAFFGate(self.out_channels) for _ in range(3)])
        self.fpn = FeaturePyramidNetwork([self.out_channels, self.out_channels, self.out_channels], self.out_channels)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feature_maps: list[torch.Tensor] = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in {3, 5, 6}:
                stage = len(feature_maps)
                feature_maps.append(self.cbam[stage](x))
        projected = [projection(feature) for projection, feature in zip(self.projection, feature_maps)]
        global_context = self.vit(projected[-1])
        fused = [gate(local, global_context) for gate, local in zip(self.ddaff, projected)]
        return self.fpn(OrderedDict((str(index), feature) for index, feature in enumerate(fused)))


def create_production_faster_rcnn(num_classes: int = NUM_DETECTION_CLASSES) -> FasterRCNN:
    backbone = ProductionCrackXNetBackbone()
    anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,)), aspect_ratios=((0.5, 1.0, 2.0),) * 3)
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2"], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=640,
        max_size=640,
    )


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
    training_config: dict[str, Any] | None = None,
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
        "training_config": training_config or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_hybrid_checkpoint(path: str | Path, device: str = "auto") -> tuple[FasterRCNN, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hybrid checkpoint not found: {path}")
    torch_device = get_device(device)
    payload = load_checkpoint_payload(path, torch_device)
    if "model_state_dict" in payload and payload.get("architecture") == "EfficientNet-B0 + CBAM + ViT + DDAFF + FPN + Faster R-CNN":
        num_classes = int(payload.get("num_classes", NUM_DETECTION_CLASSES))
        if num_classes != NUM_DETECTION_CLASSES:
            raise RuntimeError(
                f"Invalid hybrid checkpoint class count: expected {NUM_DETECTION_CLASSES}, got {num_classes}."
            )
        model = create_production_faster_rcnn(num_classes=num_classes)
        model.load_state_dict(payload["model_state_dict"])
        model.to(torch_device)
        model.eval()
        return model, payload
    if payload.get("model_mode") != "hybrid" or "hybrid_config" not in payload:
        raise RuntimeError(f"Invalid hybrid checkpoint: {path}")
    num_classes = int(payload.get("num_classes", NUM_DETECTION_CLASSES))
    if num_classes != NUM_DETECTION_CLASSES:
        raise RuntimeError(
            f"Invalid hybrid checkpoint class count: expected {NUM_DETECTION_CLASSES}, got {num_classes}."
        )
    config = hybrid_config_from_dict(payload["hybrid_config"], device=str(torch_device))
    model = create_hybrid_faster_rcnn(config, num_classes=num_classes)
    model.load_state_dict(payload["model_state"])
    model.to(torch_device)
    model.eval()
    return model, payload
