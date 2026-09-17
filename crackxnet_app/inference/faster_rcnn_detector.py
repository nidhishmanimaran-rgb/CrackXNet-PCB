from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms import functional as F

from crackxnet_app.config import CLASS_ID_TO_NAME, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DEVICE
from crackxnet_app.deeppcb.model import get_device, load_checkpoint
from crackxnet_app.schemas import BoundingBox, DefectPrediction


@dataclass
class FasterRCNNInspectionDetector:
    checkpoint_path: Path
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    device: str = DEFAULT_DEVICE
    image_size: int = 640

    def __post_init__(self) -> None:
        self.torch_device = get_device(self.device)
        self.model, self.checkpoint = load_checkpoint(self.checkpoint_path, self.torch_device, pretrained=False)
        self.model.eval()

    @property
    def status(self) -> str:
        return "DeepPCB Faster R-CNN"

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
                    rationale="Faster R-CNN prediction from the loaded DeepPCB checkpoint.",
                )
            )
        heatmap = _heatmap_from_defects((width, height), defects)
        return defects, heatmap


def _severity_from_box(bbox: BoundingBox, image_area: int, confidence: float) -> float:
    area_ratio = bbox.area / max(1, image_area)
    longest = max(bbox.width, bbox.height) / max(1, int(image_area**0.5))
    return min(1.0, 0.12 + area_ratio * 60.0 + longest * 0.35 + confidence * 0.18)


def _heatmap_from_defects(size: tuple[int, int], defects: list[DefectPrediction]) -> Image.Image:
    heat = Image.new("L", size, 0)
    draw = ImageDraw.Draw(heat)
    for defect in defects:
        value = int(80 + 175 * max(0.0, min(1.0, defect.confidence)))
        b = defect.bbox
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], fill=value)
    return heat.filter(ImageFilter.GaussianBlur(radius=16))
