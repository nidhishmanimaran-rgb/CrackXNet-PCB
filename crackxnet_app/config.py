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
    os.getenv("CRACKXNET_CHECKPOINT", PROJECT_ROOT / "outputs" / "trained" / "baseline" / "best.pth")
)
DEFAULT_DEVICE = os.getenv("CRACKXNET_DEVICE", "auto")
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CRACKXNET_CONFIDENCE", "0.5"))
DEFAULT_MODEL_MODE = os.getenv("CRACKXNET_MODEL_MODE", "baseline")
MAX_UPLOAD_BYTES = int(os.getenv("CRACKXNET_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_UPLOAD_PIXELS = int(os.getenv("CRACKXNET_MAX_UPLOAD_PIXELS", "40000000"))
REPORT_CACHE_MAX_ENTRIES = int(os.getenv("CRACKXNET_REPORT_CACHE_MAX_ENTRIES", "100"))


@dataclass(frozen=True)
class SecurityConfig:
    """Optional in-process safeguards for local/public API hosting.

    Set `CRACKXNET_API_KEY` before exposing the service outside a trusted local
    network. Rate limiting is process-local; production deployments should also
    enforce it at the reverse proxy or gateway.
    """

    api_key: str | None = os.getenv("CRACKXNET_API_KEY") or None
    rate_limit_requests: int = int(os.getenv("CRACKXNET_RATE_LIMIT_REQUESTS", "60"))
    rate_limit_window_seconds: int = int(os.getenv("CRACKXNET_RATE_LIMIT_WINDOW_SECONDS", "60"))

    def __post_init__(self) -> None:
        if self.rate_limit_requests <= 0 or self.rate_limit_window_seconds <= 0:
            raise ValueError("Security rate-limit settings must be positive.")


DEFAULT_SECURITY_CONFIG = SecurityConfig()


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


@dataclass(frozen=True)
class DDAFFConfig:
    fusion_dim: int = int(os.getenv("CRACKXNET_DDAFF_FUSION_DIM", "256"))
    dropout: float = float(os.getenv("CRACKXNET_DDAFF_DROPOUT", "0.0"))
    use_layer_norm: bool = os.getenv("CRACKXNET_DDAFF_LAYER_NORM", "1") == "1"
    device: str = os.getenv("CRACKXNET_DDAFF_DEVICE", DEFAULT_DEVICE)


DEFAULT_DDAFF_CONFIG = DDAFFConfig()


@dataclass(frozen=True)
class HybridDetectorConfig:
    enabled: bool = os.getenv("CRACKXNET_HYBRID_ENABLED", "0") == "1"
    checkpoint_path: Path = Path(
        os.getenv("CRACKXNET_HYBRID_CHECKPOINT", PROJECT_ROOT / "outputs" / "trained" / "hybrid" / "best.pth")
    )
    image_size: int = int(os.getenv("CRACKXNET_HYBRID_IMAGE_SIZE", "224"))
    fusion_dim: int = int(os.getenv("CRACKXNET_HYBRID_FUSION_DIM", "256"))
    fpn_out_channels: int = int(os.getenv("CRACKXNET_HYBRID_FPN_CHANNELS", "256"))
    local_input_size: int = int(os.getenv("CRACKXNET_HYBRID_LOCAL_SIZE", "224"))
    vit_input_size: int = int(os.getenv("CRACKXNET_HYBRID_VIT_SIZE", "224"))
    vit_variant: str = os.getenv("CRACKXNET_HYBRID_VIT_VARIANT", "vit_b_16")
    local_pretrained: bool = os.getenv("CRACKXNET_HYBRID_LOCAL_PRETRAINED", "0") == "1"
    vit_pretrained: bool = os.getenv("CRACKXNET_HYBRID_VIT_PRETRAINED", "0") == "1"
    confidence_threshold: float = float(os.getenv("CRACKXNET_HYBRID_CONFIDENCE", str(DEFAULT_CONFIDENCE_THRESHOLD)))
    device: str = os.getenv("CRACKXNET_HYBRID_DEVICE", DEFAULT_DEVICE)


DEFAULT_HYBRID_DETECTOR_CONFIG = HybridDetectorConfig()


@dataclass(frozen=True)
class SeverityConfig:
    mode: str = os.getenv("CRACKXNET_SEVERITY_MODE", "rule_based")
    low_max_score: float = float(os.getenv("CRACKXNET_SEVERITY_LOW_MAX", "0.35"))
    medium_max_score: float = float(os.getenv("CRACKXNET_SEVERITY_MEDIUM_MAX", "0.72"))
    area_weight: float = float(os.getenv("CRACKXNET_SEVERITY_AREA_WEIGHT", "45.0"))
    confidence_weight: float = float(os.getenv("CRACKXNET_SEVERITY_CONFIDENCE_WEIGHT", "0.25"))
    elongation_weight: float = float(os.getenv("CRACKXNET_SEVERITY_ELONGATION_WEIGHT", "0.12"))


DEFAULT_SEVERITY_CONFIG = SeverityConfig()


@dataclass(frozen=True)
class ExplainabilityConfig:
    enabled: bool = os.getenv("CRACKXNET_EXPLAINABILITY_ENABLED", "1") == "1"
    method: str = os.getenv("CRACKXNET_EXPLAINABILITY_METHOD", "grad_cam")
    top_k: int = int(os.getenv("CRACKXNET_EXPLAINABILITY_TOP_K", "5"))


DEFAULT_EXPLAINABILITY_CONFIG = ExplainabilityConfig()


@dataclass(frozen=True)
class QualityConfig:
    mode: str = os.getenv("CRACKXNET_QUALITY_MODE", "rule_based")
    min_confidence: float = float(os.getenv("CRACKXNET_QUALITY_MIN_CONFIDENCE", "0.25"))
    reject_on_high_severity_count: int = int(os.getenv("CRACKXNET_QUALITY_REJECT_HIGH_COUNT", "1"))
    reject_on_total_defects: int = int(os.getenv("CRACKXNET_QUALITY_REJECT_DEFECT_COUNT", "8"))
    rework_on_medium_severity_count: int = int(os.getenv("CRACKXNET_QUALITY_REWORK_MEDIUM_COUNT", "1"))
    rework_on_low_severity_count: int = int(os.getenv("CRACKXNET_QUALITY_REWORK_LOW_COUNT", "3"))
    pass_on_no_detections: bool = os.getenv("CRACKXNET_QUALITY_PASS_ON_NO_DETECTIONS", "1") == "1"


DEFAULT_QUALITY_CONFIG = QualityConfig()
