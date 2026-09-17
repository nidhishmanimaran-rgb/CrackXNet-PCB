from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from crackxnet_app.deeppcb.dataset import DeepPCBSample


class DeepPCBTorchDataset(Dataset):
    def __init__(self, samples: list[DeepPCBSample], transforms=None) -> None:
        self.samples = samples
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        boxes = torch.tensor([[a.x1, a.y1, a.x2, a.y2] for a in sample.annotations], dtype=torch.float32)
        labels = torch.tensor([a.class_id for a in sample.annotations], dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "path": str(Path(sample.image_path)),
        }
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target
