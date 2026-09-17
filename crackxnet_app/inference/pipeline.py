from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from crackxnet_app.config import DEFAULT_THRESHOLDS, InspectionThresholds
from crackxnet_app.inference.baseline_detector import BaselinePCBDetector, compose_heatmap, draw_overlay
from crackxnet_app.inference.modules import DDAFFPlaceholder, EfficientNetCBAMPlaceholder, ViTPlaceholder
from crackxnet_app.schemas import InspectionResult


@dataclass
class PipelineOutputs:
    result: InspectionResult
    overlay: Image.Image
    heatmap: Image.Image


class CrackXNetPipeline:
    def __init__(self, thresholds: InspectionThresholds = DEFAULT_THRESHOLDS) -> None:
        self.thresholds = thresholds
        self.local_features = EfficientNetCBAMPlaceholder()
        self.global_features = ViTPlaceholder()
        self.fusion = DDAFFPlaceholder()
        self.detector = BaselinePCBDetector(thresholds)

    def inspect(self, image: Image.Image, filename: str = "uploaded-image") -> PipelineOutputs:
        defects, saliency = self.detector.detect(image)
        decision = self._decision(defects)
        max_severity = max((defect.severity for defect in defects), default=0.0)
        notes = [
            "MVP baseline: heuristic detector/classifier, not trained CrackXNet accuracy.",
            "EfficientNet-B0+CBAM, ViT, DDAFF, and Faster R-CNN are modular extension points.",
        ]
        result = InspectionResult(
            filename=filename,
            image_width=image.width,
            image_height=image.height,
            decision=decision,
            max_severity=round(float(max_severity), 3),
            defects=defects,
            notes=notes,
        )
        overlay = draw_overlay(image, defects)
        heatmap = compose_heatmap(image, saliency)
        return PipelineOutputs(result=result, overlay=overlay, heatmap=heatmap)

    def _decision(self, defects: list) -> str:
        if not defects:
            return "PASS"
        max_severity = max(defect.severity for defect in defects)
        if max_severity >= self.thresholds.reject_min_severity or len(defects) >= self.thresholds.reject_defect_count:
            return "REJECT"
        if max_severity >= self.thresholds.rework_min_severity:
            return "REWORK"
        return "PASS"
