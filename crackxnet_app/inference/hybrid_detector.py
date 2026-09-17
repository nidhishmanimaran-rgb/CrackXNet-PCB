from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F

from crackxnet_app.config import CLASS_ID_TO_NAME, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DEVICE
from crackxnet_app.inference.faster_rcnn_detector import _heatmap_from_defects, _severity_from_box
from crackxnet_app.models.crackxnet_detector import load_hybrid_checkpoint
from crackxnet_app.schemas import BoundingBox, DefectPrediction


@dataclass
class HybridInspectionDetector:
    checkpoint_path: Path
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    device: str = DEFAULT_DEVICE

    def __post_init__(self) -> None:
        self.model, self.checkpoint = load_hybrid_checkpoint(self.checkpoint_path, self.device)
        self.torch_device = next(self.model.parameters()).device
        config = self.checkpoint["hybrid_config"]
        self.image_size = int(config.get("image_size", 224))
        self.model.eval()

    @property
    def status(self) -> str:
        return "CrackXNet Hybrid Detector"

    def detect(self, image: Image.Image) -> tuple[list[DefectPrediction], Image.Image]:
        original = image.convert("RGB")
        width, height = original.size
        resized = original.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tensor = F.to_tensor(resized).to(self.torch_device)
        with torch.no_grad():
            prediction = self.model([tensor])[0]
        defects: list[DefectPrediction] = []
        scale_x = width / self.image_size
        scale_y = height / self.image_size
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            confidence = float(score.detach().cpu())
            if confidence < self.confidence_threshold:
                continue
            class_id = int(label.detach().cpu())
            if class_id not in CLASS_ID_TO_NAME:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
            bbox = BoundingBox(
                x1=max(0, min(width, int(round(x1 * scale_x)))),
                y1=max(0, min(height, int(round(y1 * scale_y)))),
                x2=max(0, min(width, int(round(x2 * scale_x)))),
                y2=max(0, min(height, int(round(y2 * scale_y)))),
            )
            if bbox.area <= 0:
                continue
            severity = _severity_from_box(bbox, width * height, confidence)
            defects.append(
                DefectPrediction(
                    label=CLASS_ID_TO_NAME[class_id],
                    confidence=round(confidence, 3),
                    severity=round(severity, 3),
                    bbox=bbox,
                    rationale="Hybrid CrackXNet Faster R-CNN prediction from the loaded checkpoint.",
                )
            )
        return defects, _heatmap_from_defects((width, height), defects)
