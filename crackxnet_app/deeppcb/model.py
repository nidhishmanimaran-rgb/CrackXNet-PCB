from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from crackxnet_app.config import NUM_DETECTION_CLASSES


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def create_faster_rcnn(num_classes: int = NUM_DETECTION_CLASSES, pretrained: bool = True, image_size: int = 640):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=image_size,
        max_size=image_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    image_size: int = 640,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "num_classes": NUM_DETECTION_CLASSES,
        "model_name": "fasterrcnn_mobilenet_v3_large_fpn",
        "image_size": image_size,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu", pretrained: bool = False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_location = get_device(device) if isinstance(device, str) else device
    payload = torch.load(path, map_location=map_location)
    if "model_state" not in payload:
        raise RuntimeError(f"Invalid checkpoint: {path} does not contain model_state")
    num_classes = int(payload.get("num_classes", NUM_DETECTION_CLASSES))
    image_size = int(payload.get("image_size", 640))
    model = create_faster_rcnn(num_classes=num_classes, pretrained=pretrained, image_size=image_size)
    model.load_state_dict(payload["model_state"])
    model.to(map_location)
    model.eval()
    return model, payload
