from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from crackxnet_app.features.efficientnet_cbam import LocalFeatureOutput
from crackxnet_app.features.vit import GlobalFeatureOutput


class LocalFeatureExtractor(Protocol):
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract local PCB features."""


@runtime_checkable
class TorchLocalFeatureExtractor(Protocol):
    def extract_local_features(self, image_tensor: torch.Tensor) -> LocalFeatureOutput:
        """Extract local spatial PCB features for future fusion."""


@runtime_checkable
class TorchGlobalFeatureExtractor(Protocol):
    def extract_global_features(self, image_tensor: torch.Tensor) -> GlobalFeatureOutput:
        """Extract global contextual PCB features for future fusion."""


class GlobalFeatureExtractor(Protocol):
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract global PCB context features."""


class FeatureFusion(Protocol):
    def fuse(self, local_features: np.ndarray, global_features: np.ndarray) -> np.ndarray:
        """Fuse local and global features."""


@dataclass
class EfficientNetCBAMPlaceholder:
    """Extension point for EfficientNet-B0 + CBAM local features."""

    def extract(self, image: np.ndarray) -> np.ndarray:
        return image


@dataclass
class ViTPlaceholder:
    """Extension point for ViT global features."""

    def extract(self, image: np.ndarray) -> np.ndarray:
        return image.mean(axis=(0, 1), keepdims=True)


@dataclass
class DDAFFPlaceholder:
    """Extension point for Dynamic Defect-Aware Adaptive Feature Fusion."""

    def fuse(self, local_features: np.ndarray, global_features: np.ndarray) -> np.ndarray:
        return local_features
