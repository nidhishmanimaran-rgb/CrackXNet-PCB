from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32
from torchvision.transforms import functional as F

from crackxnet_app.config import DEFAULT_GLOBAL_FEATURE_CONFIG, GlobalFeatureConfig
from crackxnet_app.deeppcb.model import get_device
from crackxnet_app.features.cbam import validate_feature_tensor


@dataclass(frozen=True)
class ViTVariantSpec:
    builder: object
    weights_enum: object
    patch_size: int


VIT_VARIANTS = {
    "vit_b_16": ViTVariantSpec(builder=vit_b_16, weights_enum=ViT_B_16_Weights, patch_size=16),
    "vit_b_32": ViTVariantSpec(builder=vit_b_32, weights_enum=ViT_B_32_Weights, patch_size=32),
}


@dataclass(frozen=True)
class GlobalFeatureOutput:
    """ViT global feature representation for future DDAFF integration.

    `class_token`: contextual CLS token, shape `(batch, hidden_dim)`.
    `patch_tokens`: contextual patch tokens, shape `(batch, num_patches, hidden_dim)`.
    `pooled_features`: mean-pooled patch representation, shape `(batch, hidden_dim)`.
    """

    class_token: torch.Tensor
    patch_tokens: torch.Tensor
    pooled_features: torch.Tensor
    backbone_name: str
    input_shape: tuple[int, ...]
    patch_size: int


class ViTFeatureBackbone(nn.Module):
    """Torchvision ViT backbone that returns encoder tokens, not classifier logits."""

    def __init__(self, variant: str = "vit_b_16", pretrained: bool = False, image_size: int = 224) -> None:
        super().__init__()
        if variant not in VIT_VARIANTS:
            supported = ", ".join(sorted(VIT_VARIANTS))
            raise ValueError(f"Unsupported ViT variant {variant!r}. Supported variants: {supported}.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        spec = VIT_VARIANTS[variant]
        if image_size % spec.patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={spec.patch_size}.")
        weights = spec.weights_enum.DEFAULT if pretrained else None
        self.model = spec.builder(weights=weights, image_size=image_size)
        self.variant = variant
        self.patch_size = spec.patch_size
        self.hidden_dim = self.model.hidden_dim
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        validate_feature_tensor(x, module_name=self.__class__.__name__)
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"{self.__class__.__name__} expected {self.image_size}x{self.image_size} input, "
                f"got {tuple(x.shape[-2:])}."
            )
        x = self.model._process_input(x)
        batch_size = x.shape[0]
        class_token = self.model.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([class_token, x], dim=1)
        encoded = self.model.encoder(tokens)
        return encoded[:, 0], encoded[:, 1:]


class ViTGlobalFeatureExtractor(nn.Module):
    """Vision Transformer global/contextual PCB feature extractor."""

    def __init__(
        self,
        config: GlobalFeatureConfig | None = None,
        *,
        variant: str | None = None,
        pretrained: bool | None = None,
        input_size: int | None = None,
        patch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        base_config = config or DEFAULT_GLOBAL_FEATURE_CONFIG
        self.variant = variant or base_config.vit_variant
        self.input_size = input_size if input_size is not None else base_config.input_size
        self.device = get_device(device or base_config.device)
        use_pretrained = base_config.vit_pretrained if pretrained is None else pretrained
        expected_patch = VIT_VARIANTS[self.variant].patch_size if self.variant in VIT_VARIANTS else None
        configured_patch = patch_size if patch_size is not None else base_config.patch_size
        if configured_patch is not None and expected_patch is not None and configured_patch != expected_patch:
            raise ValueError(
                f"{self.variant} uses patch_size={expected_patch}; received patch_size={configured_patch}."
            )
        self.backbone = ViTFeatureBackbone(self.variant, pretrained=use_pretrained, image_size=self.input_size)
        self.to(self.device)
        self.eval()

    @classmethod
    def from_config(cls, config: GlobalFeatureConfig = DEFAULT_GLOBAL_FEATURE_CONFIG) -> "ViTGlobalFeatureExtractor":
        extractor = cls(config=config)
        if config.checkpoint_path:
            extractor.load_feature_checkpoint(config.checkpoint_path)
        return extractor

    def forward(self, image_tensor: torch.Tensor) -> GlobalFeatureOutput:
        validate_feature_tensor(image_tensor, module_name=self.__class__.__name__)
        image_tensor = image_tensor.to(self.device)
        class_token, patch_tokens = self.backbone(image_tensor)
        pooled = patch_tokens.mean(dim=1)
        return GlobalFeatureOutput(
            class_token=class_token,
            patch_tokens=patch_tokens,
            pooled_features=pooled,
            backbone_name=self.variant,
            input_shape=tuple(image_tensor.shape),
            patch_size=self.backbone.patch_size,
        )

    def extract_global_features(self, image_tensor: torch.Tensor) -> GlobalFeatureOutput:
        with torch.no_grad():
            return self.forward(image_tensor)

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        return F.to_tensor(image).unsqueeze(0)

    def extract_from_pil(self, image: Image.Image) -> GlobalFeatureOutput:
        return self.extract_global_features(self.preprocess_pil(image))

    def load_feature_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"ViT feature checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        self.load_state_dict(state)
