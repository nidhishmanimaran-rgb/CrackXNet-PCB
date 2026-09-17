from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from crackxnet_app.config import DDAFFConfig, DEFAULT_DDAFF_CONFIG
from crackxnet_app.deeppcb.model import get_device
from crackxnet_app.features.cbam import validate_feature_tensor


@dataclass(frozen=True)
class DDAFFConfigSnapshot:
    fusion_dim: int
    dropout: float
    use_layer_norm: bool
    device: str


@dataclass(frozen=True)
class DDAFFOutput:
    """DDAFF fused representation for future detector integration.

    `fused_map`: spatial fused tensor, shape `(batch, fusion_dim, height, width)`.
    `pooled_features`: global average pooled fused vector, shape `(batch, fusion_dim)`.
    `fusion_weights`: adaptive stream weights, shape `(batch, 2)`, ordered local/global.
    `local_projected`: projected local map, shape `(batch, fusion_dim, height, width)`.
    `global_projected`: projected global map, shape `(batch, fusion_dim, height, width)`.
    """

    fused_map: torch.Tensor
    pooled_features: torch.Tensor
    fusion_weights: torch.Tensor
    local_projected: torch.Tensor
    global_projected: torch.Tensor


class DynamicDefectAwareFeatureFusion(nn.Module):
    """Trainable adaptive fusion for local EfficientNet-CBAM and global ViT features.

    The module accepts a local NCHW feature map and ViT patch tokens. Both streams
    are projected to `fusion_dim`, aligned to the local spatial grid, weighted by
    learned adaptive gates, then combined into a detector-ready spatial map.
    """

    def __init__(
        self,
        local_channels: int,
        global_dim: int,
        config: DDAFFConfig | None = None,
        *,
        fusion_dim: int | None = None,
        dropout: float | None = None,
        use_layer_norm: bool | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        base_config = config or DEFAULT_DDAFF_CONFIG
        self.config_snapshot = DDAFFConfigSnapshot(
            fusion_dim=fusion_dim if fusion_dim is not None else base_config.fusion_dim,
            dropout=dropout if dropout is not None else base_config.dropout,
            use_layer_norm=use_layer_norm if use_layer_norm is not None else base_config.use_layer_norm,
            device=device or base_config.device,
        )
        if local_channels <= 0:
            raise ValueError("local_channels must be positive.")
        if global_dim <= 0:
            raise ValueError("global_dim must be positive.")
        if self.config_snapshot.fusion_dim <= 0:
            raise ValueError("fusion_dim must be positive.")
        if not 0.0 <= self.config_snapshot.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        dim = self.config_snapshot.fusion_dim
        self.local_projection = nn.Conv2d(local_channels, dim, kernel_size=1, bias=False)
        self.global_projection = nn.Linear(global_dim, dim, bias=False)
        self.local_score = nn.Linear(dim, 1)
        self.global_score = nn.Linear(dim, 1)
        self.stream_logits = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout2d(self.config_snapshot.dropout)
        self.norm = nn.GroupNorm(1, dim) if self.config_snapshot.use_layer_norm else nn.Identity()
        self.device = get_device(self.config_snapshot.device)
        self.to(self.device)

    def forward(self, local_feature_map: torch.Tensor, global_patch_tokens: torch.Tensor) -> DDAFFOutput:
        validate_feature_tensor(local_feature_map, module_name=self.__class__.__name__)
        self._validate_patch_tokens(global_patch_tokens)
        if local_feature_map.shape[0] != global_patch_tokens.shape[0]:
            raise ValueError(
                "local_feature_map and global_patch_tokens must have matching batch size, "
                f"got {local_feature_map.shape[0]} and {global_patch_tokens.shape[0]}."
            )

        local_feature_map = local_feature_map.to(self.device)
        global_patch_tokens = global_patch_tokens.to(self.device)
        local_projected = self.local_projection(local_feature_map)
        global_map = self._tokens_to_map(global_patch_tokens)
        global_projected = self.global_projection(global_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if global_projected.shape[-2:] != local_projected.shape[-2:]:
            global_projected = F.interpolate(
                global_projected,
                size=local_projected.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        local_context = local_projected.mean(dim=(2, 3))
        global_context = global_projected.mean(dim=(2, 3))
        scores = torch.cat([self.local_score(local_context), self.global_score(global_context)], dim=1)
        scores = scores + self.stream_logits.unsqueeze(0)
        fusion_weights = torch.softmax(scores, dim=1)
        fused = (
            fusion_weights[:, 0].view(-1, 1, 1, 1) * local_projected
            + fusion_weights[:, 1].view(-1, 1, 1, 1) * global_projected
        )
        fused = self.norm(self.dropout(fused))
        pooled = fused.mean(dim=(2, 3))
        return DDAFFOutput(
            fused_map=fused,
            pooled_features=pooled,
            fusion_weights=fusion_weights,
            local_projected=local_projected,
            global_projected=global_projected,
        )

    def diagnostics(self, output: DDAFFOutput) -> dict[str, object]:
        return {
            "local_projected_shape": tuple(output.local_projected.shape),
            "global_projected_shape": tuple(output.global_projected.shape),
            "fusion_weights": output.fusion_weights.detach().cpu().tolist(),
            "fused_shape": tuple(output.fused_map.shape),
        }

    @staticmethod
    def _validate_patch_tokens(tokens: torch.Tensor) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("global_patch_tokens must be a torch.Tensor.")
        if tokens.ndim != 3:
            raise ValueError(f"global_patch_tokens must be BNC, got shape {tuple(tokens.shape)}.")
        if tokens.shape[1] <= 0 or tokens.shape[2] <= 0:
            raise ValueError("global_patch_tokens must contain at least one token and one channel.")
        root = sqrt(tokens.shape[1])
        if not isclose(root, round(root)):
            raise ValueError(
                "global_patch_tokens token count must be a square number so tokens can be reshaped to a grid."
            )

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, channels = tokens.shape
        side = int(sqrt(token_count))
        return tokens.reshape(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()
