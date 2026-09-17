from __future__ import annotations

from PIL import Image, ImageOps
import numpy as np


def load_rgb_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(__import__("io").BytesIO(file_bytes))
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc


def resize_for_inference(image: Image.Image, max_side: int = 1280) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def normalize_image(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr
