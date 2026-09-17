"""Feature extractors for staged CrackXNet components."""

from crackxnet_app.features.efficientnet_cbam import EfficientNetCBAMLocalFeatureExtractor, LocalFeatureOutput
from crackxnet_app.features.ddaff import DDAFFConfigSnapshot, DDAFFOutput, DynamicDefectAwareFeatureFusion
from crackxnet_app.features.vit import GlobalFeatureOutput, ViTGlobalFeatureExtractor

__all__ = [
    "DDAFFConfigSnapshot",
    "DDAFFOutput",
    "DynamicDefectAwareFeatureFusion",
    "EfficientNetCBAMLocalFeatureExtractor",
    "GlobalFeatureOutput",
    "LocalFeatureOutput",
    "ViTGlobalFeatureExtractor",
]
