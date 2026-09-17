from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFECT_CLASSES = (
    "Open Circuit",
    "Short Circuit",
    "Mouse Bite",
    "Spur",
    "Spurious Copper",
    "Pin Hole",
)

CLASS_ID_TO_NAME = {index: name for index, name in enumerate(DEFECT_CLASSES, start=1)}
CLASS_NAME_TO_ID = {name: index for index, name in CLASS_ID_TO_NAME.items()}
BACKGROUND_CLASS_ID = 0
NUM_DETECTION_CLASSES = len(DEFECT_CLASSES) + 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.getenv("DEEPCB_DATA_ROOT", PROJECT_ROOT / "data" / "DeepPCB"))
DEFAULT_CHECKPOINT_PATH = Path(
    os.getenv("CRACKXNET_CHECKPOINT", PROJECT_ROOT / "outputs" / "checkpoints" / "best.pth")
)
DEFAULT_DEVICE = os.getenv("CRACKXNET_DEVICE", "auto")
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CRACKXNET_CONFIDENCE", "0.5"))


@dataclass(frozen=True)
class InspectionThresholds:
    min_component_area_ratio: float = 0.00005
    max_component_area_ratio: float = 0.2
    rework_min_severity: float = 0.35
    reject_min_severity: float = 0.72
    reject_defect_count: int = 8
    heatmap_blur_radius: int = 12


DEFAULT_THRESHOLDS = InspectionThresholds()
