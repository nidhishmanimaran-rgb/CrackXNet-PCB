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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class InspectionResult:
    filename: str
    image_width: int
    image_height: int
    decision: str
    max_severity: float
    defects: list[DefectPrediction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "decision": self.decision,
            "max_severity": self.max_severity,
            "defects": [defect.to_dict() for defect in self.defects],
            "notes": list(self.notes),
        }
