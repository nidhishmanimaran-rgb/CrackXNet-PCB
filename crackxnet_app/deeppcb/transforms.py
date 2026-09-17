from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F


@dataclass
class TransformConfig:
    image_size: int = 640
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotate90_prob: float = 0.0
    random_crop_prob: float = 0.0
    color_jitter_prob: float = 0.2
    brightness: float = 0.12
    contrast: float = 0.12
    saturation: float = 0.08


class DeepPCBTransforms:
    """BBox-safe transforms for DeepPCB.

    Validation/inference only resize and convert to tensor. Training may add
    flips, 90-degree rotations, conservative crops, and color jitter.
    """

    def __init__(self, config: TransformConfig | None = None, train: bool = False) -> None:
        self.config = config or TransformConfig()
        self.train = train

    def __call__(self, image: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        image = image.convert("RGB")
        boxes = target["boxes"].clone().float()
        labels = target["labels"].clone().long()

        if self.train:
            image, boxes = self._augment(image, boxes)

        image, boxes = self._resize(image, boxes)
        target = dict(target)
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        return F.to_tensor(image), target

    def _augment(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        width, height = image.size
        if random.random() < self.config.horizontal_flip_prob:
            image = F.hflip(image)
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = x1, x2
        if random.random() < self.config.vertical_flip_prob:
            image = F.vflip(image)
            y1 = height - boxes[:, 3]
            y2 = height - boxes[:, 1]
            boxes[:, 1], boxes[:, 3] = y1, y2
        if random.random() < self.config.rotate90_prob:
            image = image.rotate(90, expand=True)
            old = boxes.clone()
            boxes[:, 0] = old[:, 1]
            boxes[:, 1] = width - old[:, 2]
            boxes[:, 2] = old[:, 3]
            boxes[:, 3] = width - old[:, 0]
            width, height = image.size
        if random.random() < self.config.random_crop_prob:
            image, boxes = self._safe_crop(image, boxes)
        if random.random() < self.config.color_jitter_prob:
            image = self._color_jitter(image)
        return image, boxes

    def _safe_crop(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        width, height = image.size
        margin = min(width, height) // 20
        if margin <= 0:
            return image, boxes
        left = random.randint(0, margin)
        top = random.randint(0, margin)
        right = width - random.randint(0, margin)
        bottom = height - random.randint(0, margin)
        clipped = boxes.clone()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clamp(left, right)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clamp(top, bottom)
        keep = ((clipped[:, 2] - clipped[:, 0]) > 1) & ((clipped[:, 3] - clipped[:, 1]) > 1)
        if not torch.all(keep):
            return image, boxes
        cropped = image.crop((left, top, right, bottom))
        clipped[:, [0, 2]] -= left
        clipped[:, [1, 3]] -= top
        return cropped, clipped

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        c = self.config
        brightness = 1.0 + random.uniform(-c.brightness, c.brightness)
        contrast = 1.0 + random.uniform(-c.contrast, c.contrast)
        saturation = 1.0 + random.uniform(-c.saturation, c.saturation)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def _resize(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        size = self.config.image_size
        old_w, old_h = image.size
        if old_w == size and old_h == size:
            return image, boxes
        scale_x = size / old_w
        scale_y = size / old_h
        resized = image.resize((size, size), Image.Resampling.BILINEAR)
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        return resized, boxes


def collate_fn(batch):
    return tuple(zip(*batch))
