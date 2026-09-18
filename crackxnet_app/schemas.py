from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class DefectPrediction:
    label: str
    confidence: float
    severity: float
    bbox: BoundingBox
    rationale: str
    severity_label: str = "LOW"
    severity_reason: str = "Rule-based severity has not been evaluated."
    explanation_region: BoundingBox | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        data["explanation_region"] = self.explanation_region.to_dict() if self.explanation_region else None
        return data


@dataclass
class QualityAssessment:
    status: str
    reason: str
    defect_count: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    detected_defect_types: list[str]
    confidence_summary: dict[str, float | int | None]
    recommendation: str
    criteria: list[str]
    ignored_low_confidence_count: int = 0
    mode: str = "rule_based"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InspectionResult:
    filename: str
    image_width: int
    image_height: int
    decision: str
    max_severity: float
    defects: list[DefectPrediction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    quality: QualityAssessment | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "decision": self.decision,
            "max_severity": self.max_severity,
            "defects": [defect.to_dict() for defect in self.defects],
            "notes": list(self.notes),
            "quality": self.quality.to_dict() if self.quality else None,
            "model_metadata": dict(self.model_metadata),
        }
