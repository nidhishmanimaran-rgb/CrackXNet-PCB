from __future__ import annotations

from dataclasses import dataclass


DEFECT_CLASSES = (
    "Open Circuit",
    "Short Circuit",
    "Mouse Bite",
    "Spur",
    "Pin Hole",
    "Spurious Copper",
)


@dataclass(frozen=True)
class InspectionThresholds:
    min_component_area_ratio: float = 0.00005
    max_component_area_ratio: float = 0.2
    rework_min_severity: float = 0.35
    reject_min_severity: float = 0.72
    reject_defect_count: int = 8
    heatmap_blur_radius: int = 12


DEFAULT_THRESHOLDS = InspectionThresholds()
