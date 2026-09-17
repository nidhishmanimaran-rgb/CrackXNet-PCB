from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from crackxnet_app.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DEVICE, DEFAULT_EXPLAINABILITY_CONFIG, DEFAULT_HYBRID_DETECTOR_CONFIG, DEFAULT_MODEL_MODE, DEFAULT_QUALITY_CONFIG, DEFAULT_SEVERITY_CONFIG, DEFAULT_THRESHOLDS, InspectionThresholds
from crackxnet_app.explainability import GradCAMExplainer
from crackxnet_app.inference.baseline_detector import BaselinePCBDetector, compose_heatmap, draw_overlay
from crackxnet_app.inference.faster_rcnn_detector import FasterRCNNInspectionDetector
from crackxnet_app.inference.hybrid_detector import HybridInspectionDetector
from crackxnet_app.schemas import InspectionResult
from crackxnet_app.quality import RuleBasedQualityAssessor
from crackxnet_app.severity import RuleBasedSeverityEstimator


@dataclass
class PipelineOutputs:
    result: InspectionResult
    overlay: Image.Image
    heatmap: Image.Image


class DetectorUnavailableError(RuntimeError):
    """Raised when a configured checkpoint cannot be safely initialized."""


class CrackXNetPipeline:
    def __init__(
        self,
        thresholds: InspectionThresholds = DEFAULT_THRESHOLDS,
        checkpoint_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        device: str = DEFAULT_DEVICE,
        force_demo: bool = False,
        model_mode: str = DEFAULT_MODEL_MODE,
    ) -> None:
        self.thresholds = thresholds
        self.severity_estimator = RuleBasedSeverityEstimator(DEFAULT_SEVERITY_CONFIG)
        self.quality_assessor = RuleBasedQualityAssessor(DEFAULT_QUALITY_CONFIG)
        self.explainer = GradCAMExplainer(DEFAULT_EXPLAINABILITY_CONFIG)
        self.model_status = "Baseline / Demo Mode"
        if model_mode not in {"baseline", "hybrid"}:
            raise ValueError("model_mode must be 'baseline' or 'hybrid'.")
        self.model_mode = model_mode
        default_checkpoint = DEFAULT_HYBRID_DETECTOR_CONFIG.checkpoint_path if model_mode == "hybrid" else DEFAULT_CHECKPOINT_PATH
        selected_confidence = (
            DEFAULT_HYBRID_DETECTOR_CONFIG.confidence_threshold
            if confidence_threshold is None and model_mode == "hybrid"
            else DEFAULT_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint
        self.detector = BaselinePCBDetector(thresholds)
        self.initialization_error = False
        if not force_demo and self.checkpoint_path and self.checkpoint_path.exists():
            try:
                if model_mode == "hybrid":
                    self.detector = HybridInspectionDetector(
                        checkpoint_path=self.checkpoint_path,
                        confidence_threshold=selected_confidence,
                        device=device,
                    )
                else:
                    self.detector = FasterRCNNInspectionDetector(
                        checkpoint_path=self.checkpoint_path,
                        confidence_threshold=selected_confidence,
                        device=device,
                    )
                self.model_status = self.detector.status
            except Exception:
                self.initialization_error = True
                self.model_status = "Configured detector unavailable"

    def inspect(self, image: Image.Image, filename: str = "uploaded-image") -> PipelineOutputs:
        if self.initialization_error:
            raise DetectorUnavailableError("Configured detector could not be initialized.")
        defects, saliency = self.detector.detect(image)
        defects = self.severity_estimator.apply(defects, image.size)
        quality = self.quality_assessor.assess(defects)
        explanation_note = "Demo saliency from baseline anomaly mask."
        if hasattr(self.detector, "model"):
            explanation = self.explainer.explain_detector(self.detector, image, defects)
            saliency = explanation.normalized_heatmap
            explanation_note = f"{explanation.method}: target_layer={explanation.target_layer}; {explanation.note}"
        decision = quality.status
        max_severity = max((defect.severity for defect in defects), default=0.0)
        notes = [
            f"Model status: {self.model_status}.",
            f"Severity mode: {self.severity_estimator.config.mode} transparent rule-based reasoning.",
            f"Quality mode: {quality.mode}; {quality.reason}",
            f"Explainability: {explanation_note}",
            "Hybrid mode loads CrackXNet EfficientNet-CBAM + ViT + DDAFF + FPN + Faster R-CNN checkpoints.",
            "Fallback/demo mode is heuristic and is not a trained CrackXNet model.",
        ]
        result = InspectionResult(
            filename=filename,
            image_width=image.width,
            image_height=image.height,
            decision=decision,
            max_severity=round(float(max_severity), 3),
            defects=defects,
            notes=notes,
            quality=quality,
        )
        overlay = draw_overlay(image, defects)
        heatmap = compose_heatmap(image, saliency)
        return PipelineOutputs(result=result, overlay=overlay, heatmap=heatmap)
