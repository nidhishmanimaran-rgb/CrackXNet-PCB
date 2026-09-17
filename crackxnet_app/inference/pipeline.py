from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from crackxnet_app.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DEVICE, DEFAULT_THRESHOLDS, InspectionThresholds
from crackxnet_app.inference.baseline_detector import BaselinePCBDetector, compose_heatmap, draw_overlay
from crackxnet_app.inference.faster_rcnn_detector import FasterRCNNInspectionDetector
from crackxnet_app.inference.modules import DDAFFPlaceholder, EfficientNetCBAMPlaceholder, ViTPlaceholder
from crackxnet_app.schemas import InspectionResult


@dataclass
class PipelineOutputs:
    result: InspectionResult
    overlay: Image.Image
    heatmap: Image.Image


class CrackXNetPipeline:
    def __init__(
        self,
        thresholds: InspectionThresholds = DEFAULT_THRESHOLDS,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str = DEFAULT_DEVICE,
        force_demo: bool = False,
    ) -> None:
        self.thresholds = thresholds
        self.local_features = EfficientNetCBAMPlaceholder()
        self.global_features = ViTPlaceholder()
        self.fusion = DDAFFPlaceholder()
        self.model_status = "Baseline / Demo Mode"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.detector = BaselinePCBDetector(thresholds)
        if not force_demo and self.checkpoint_path and self.checkpoint_path.exists():
            self.detector = FasterRCNNInspectionDetector(
                checkpoint_path=self.checkpoint_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
            self.model_status = self.detector.status

    def inspect(self, image: Image.Image, filename: str = "uploaded-image") -> PipelineOutputs:
        defects, saliency = self.detector.detect(image)
        decision = self._decision(defects)
        max_severity = max((defect.severity for defect in defects), default=0.0)
        notes = [
            f"Model status: {self.model_status}.",
            "Phase 2 baseline: DeepPCB Faster R-CNN when a checkpoint is loaded; heuristic demo fallback otherwise.",
            "EfficientNet-B0+CBAM, ViT, and DDAFF are not implemented in Phase 2.",
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
