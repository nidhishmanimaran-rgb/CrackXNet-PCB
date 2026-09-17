from __future__ import annotations

import torch
from torch import nn


def validate_feature_tensor(tensor: torch.Tensor, *, module_name: str = "CBAM") -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{module_name} expected a torch.Tensor input.")
    if tensor.ndim != 4:
        raise ValueError(f"{module_name} expects NCHW tensor input, got shape {tuple(tensor.shape)}.")
    if tensor.shape[1] <= 0:
        raise ValueError(f"{module_name} requires at least one channel.")


class ChannelAttention(nn.Module):
    """CBAM channel attention for NCHW feature maps."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive.")
        hidden_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_feature_tensor(x, module_name=self.__class__.__name__)
        attention = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return self.activation(attention)


class SpatialAttention(nn.Module):
    """CBAM spatial attention for NCHW feature maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError("kernel_size must be 3 or 7.")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_feature_tensor(x, module_name=self.__class__.__name__)
        avg_projection = torch.mean(x, dim=1, keepdim=True)
        max_projection, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_projection, max_projection], dim=1)
        return self.activation(self.conv(attention))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Input and output shapes are identical: `(batch, channels, height, width)`.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_feature_tensor(x, module_name=self.__class__.__name__)
        refined = x * self.channel_attention(x)
        refined = refined * self.spatial_attention(refined)
        return refined
