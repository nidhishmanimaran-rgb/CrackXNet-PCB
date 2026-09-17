from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import functional as F

from crackxnet_app.config import DEFAULT_LOCAL_FEATURE_CONFIG, LocalFeatureConfig
from crackxnet_app.deeppcb.model import get_device
from crackxnet_app.features.cbam import CBAM, validate_feature_tensor


@dataclass(frozen=True)
class LocalFeatureOutput:
    """Local feature representation reserved for future DDAFF integration.

    `feature_map`: CBAM-refined spatial feature tensor with shape
    `(batch, 1280, ceil(input_h / 32), ceil(input_w / 32))` for EfficientNet-B0.
    `pooled_features`: global average pooled tensor with shape `(batch, 1280)`.
    """

    feature_map: torch.Tensor
    pooled_features: torch.Tensor
    backbone_name: str
    input_shape: tuple[int, ...]


class EfficientNetB0FeatureBackbone(nn.Module):
    """EfficientNet-B0 without the classification head."""

    output_channels = 1280

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        self.features = model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_feature_tensor(x, module_name=self.__class__.__name__)
        return self.features(x)


class EfficientNetCBAMLocalFeatureExtractor(nn.Module):
    """EfficientNet-B0 local features refined by CBAM attention."""

    def __init__(
        self,
        config: LocalFeatureConfig | None = None,
        *,
        pretrained: bool | None = None,
        input_size: int | None = None,
        cbam_reduction_ratio: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        base_config = config or DEFAULT_LOCAL_FEATURE_CONFIG
        self.input_size = input_size if input_size is not None else base_config.input_size
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        self.device = get_device(device or base_config.device)
        use_pretrained = base_config.efficientnet_pretrained if pretrained is None else pretrained
        reduction = cbam_reduction_ratio if cbam_reduction_ratio is not None else base_config.cbam_reduction_ratio
        self.backbone = EfficientNetB0FeatureBackbone(pretrained=use_pretrained)
        self.cbam = CBAM(self.backbone.output_channels, reduction_ratio=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to(self.device)
        self.eval()

    @classmethod
    def from_config(cls, config: LocalFeatureConfig = DEFAULT_LOCAL_FEATURE_CONFIG) -> "EfficientNetCBAMLocalFeatureExtractor":
        extractor = cls(config=config)
        if config.checkpoint_path:
            extractor.load_feature_checkpoint(config.checkpoint_path)
        return extractor

    def forward(self, image_tensor: torch.Tensor) -> LocalFeatureOutput:
        validate_feature_tensor(image_tensor, module_name=self.__class__.__name__)
        image_tensor = image_tensor.to(self.device)
        feature_map = self.backbone(image_tensor)
        refined = self.cbam(feature_map)
        pooled = self.pool(refined).flatten(1)
        return LocalFeatureOutput(
            feature_map=refined,
            pooled_features=pooled,
            backbone_name="efficientnet_b0_cbam",
            input_shape=tuple(image_tensor.shape),
        )

    def extract_local_features(self, image_tensor: torch.Tensor) -> LocalFeatureOutput:
        with torch.no_grad():
            return self.forward(image_tensor)

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        tensor = F.to_tensor(image)
        return tensor.unsqueeze(0)

    def extract_from_pil(self, image: Image.Image) -> LocalFeatureOutput:
        return self.extract_local_features(self.preprocess_pil(image))

    def load_feature_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Local feature checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        self.load_state_dict(state)
