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


@dataclass(frozen=True)
class LocalFeatureConfig:
    efficientnet_pretrained: bool = os.getenv("CRACKXNET_EFFICIENTNET_PRETRAINED", "0") == "1"
    input_size: int = int(os.getenv("CRACKXNET_LOCAL_FEATURE_SIZE", "224"))
    cbam_reduction_ratio: int = int(os.getenv("CRACKXNET_CBAM_REDUCTION", "16"))
    device: str = os.getenv("CRACKXNET_LOCAL_FEATURE_DEVICE", DEFAULT_DEVICE)
    checkpoint_path: Path | None = (
        Path(os.environ["CRACKXNET_LOCAL_FEATURE_CHECKPOINT"])
        if os.getenv("CRACKXNET_LOCAL_FEATURE_CHECKPOINT")
        else None
    )


DEFAULT_LOCAL_FEATURE_CONFIG = LocalFeatureConfig()


@dataclass(frozen=True)
class GlobalFeatureConfig:
    vit_variant: str = os.getenv("CRACKXNET_VIT_VARIANT", "vit_b_16")
    vit_pretrained: bool = os.getenv("CRACKXNET_VIT_PRETRAINED", "0") == "1"
    input_size: int = int(os.getenv("CRACKXNET_VIT_INPUT_SIZE", "224"))
    patch_size: int | None = (
        int(os.environ["CRACKXNET_VIT_PATCH_SIZE"])
        if os.getenv("CRACKXNET_VIT_PATCH_SIZE")
        else None
    )
    device: str = os.getenv("CRACKXNET_VIT_DEVICE", DEFAULT_DEVICE)
    checkpoint_path: Path | None = (
        Path(os.environ["CRACKXNET_VIT_CHECKPOINT"])
        if os.getenv("CRACKXNET_VIT_CHECKPOINT")
        else None
    )


DEFAULT_GLOBAL_FEATURE_CONFIG = GlobalFeatureConfig()
