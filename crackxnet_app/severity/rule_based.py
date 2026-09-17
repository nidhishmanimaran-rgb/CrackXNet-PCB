from __future__ import annotations

from dataclasses import dataclass

from crackxnet_app.config import DEFAULT_SEVERITY_CONFIG, SeverityConfig
from crackxnet_app.schemas import BoundingBox, DefectPrediction


@dataclass(frozen=True)
class SeverityResult:
    score: float
    label: str
    reason: str
    mode: str


class RuleBasedSeverityEstimator:
    """Transparent deterministic severity head.

    This is not a learned classifier. It combines actual detection confidence,
    bounding-box area ratio, and elongation into a documented engineering score.
    """

    def __init__(self, config: SeverityConfig = DEFAULT_SEVERITY_CONFIG) -> None:
        if config.mode not in {"rule_based", "learned_ready"}:
            raise ValueError("Severity mode must be 'rule_based' or 'learned_ready'.")
        if not 0 <= config.low_max_score <= config.medium_max_score <= 1:
            raise ValueError("Severity thresholds must satisfy 0 <= low <= medium <= 1.")
        self.config = config

    def estimate(self, defect: DefectPrediction, image_size: tuple[int, int]) -> SeverityResult:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive width and height.")
        if defect.bbox.area <= 0:
            raise ValueError("defect bbox must have positive area.")
        area_ratio = defect.bbox.area / float(width * height)
        elongation = max(defect.bbox.width / max(1, defect.bbox.height), defect.bbox.height / max(1, defect.bbox.width))
        score = min(
            1.0,
            area_ratio * self.config.area_weight
            + defect.confidence * self.config.confidence_weight
            + min(0.25, elongation / 50.0) * self.config.elongation_weight
        )
        label = self._label(score)
        reason = (
            f"{self.config.mode}: area_ratio={area_ratio:.5f}, confidence={defect.confidence:.3f}, "
            f"elongation={elongation:.2f}."
        )
        return SeverityResult(score=round(float(score), 3), label=label, reason=reason, mode=self.config.mode)

    def apply(self, defects: list[DefectPrediction], image_size: tuple[int, int]) -> list[DefectPrediction]:
        enriched: list[DefectPrediction] = []
        for defect in defects:
            result = self.estimate(defect, image_size)
            enriched.append(
                DefectPrediction(
                    label=defect.label,
                    confidence=defect.confidence,
                    severity=result.score,
                    severity_label=result.label,
                    severity_reason=result.reason,
                    bbox=defect.bbox,
                    rationale=defect.rationale,
                    explanation_region=defect.explanation_region or BoundingBox(
                        defect.bbox.x1, defect.bbox.y1, defect.bbox.x2, defect.bbox.y2
                    ),
                )
            )
        return enriched

    def _label(self, score: float) -> str:
        if score <= self.config.low_max_score:
            return "LOW"
        if score <= self.config.medium_max_score:
            return "MEDIUM"
        return "HIGH"
