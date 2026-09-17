from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from PIL import Image, ImageDraw

from crackxnet_app.config import CLASS_ID_TO_NAME, DEFAULT_DATA_ROOT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DeepPCBError(ValueError):
    """Raised when the DeepPCB dataset is missing or malformed."""


@dataclass(frozen=True)
class Annotation:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int


@dataclass(frozen=True)
class DeepPCBSample:
    image_path: Path
    annotation_path: Path
    annotations: tuple[Annotation, ...]
    width: int
    height: int


def resolve_dataset_root(data_root: str | Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    if root.name == "PCBData":
        return root.parent
    return root


def pcbdata_dir(data_root: str | Path | None = None) -> Path:
    root = resolve_dataset_root(data_root)
    if (root / "PCBData").is_dir():
        return root / "PCBData"
    if root.name == "PCBData" and root.is_dir():
        return root
    raise DeepPCBError(
        f"DeepPCB PCBData directory not found under {root}. "
        "Pass --data-root pointing to DeepPCB-master or a directory containing PCBData."
    )


def split_file_paths(data_root: str | Path | None = None) -> tuple[Path, Path]:
    pcb = pcbdata_dir(data_root)
    trainval = pcb / "trainval.txt"
    test = pcb / "test.txt"
    missing = [str(path) for path in (trainval, test) if not path.exists()]
    if missing:
        raise DeepPCBError(f"Missing DeepPCB split file(s): {', '.join(missing)}")
    return trainval, test


def read_split_pairs(data_root: str | Path | None = None, split_file: str = "trainval.txt") -> list[tuple[Path, Path]]:
    pcb = pcbdata_dir(data_root)
    path = pcb / split_file
    if not path.exists():
        raise DeepPCBError(f"Missing split file: {path}")

    pairs: list[tuple[Path, Path]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 2:
            raise DeepPCBError(f"{path}:{line_no}: expected '<image> <annotation>', got: {raw!r}")
        image_path = _resolve_image_path(pcb, parts[0], path, line_no)
        annotation_path = pcb / parts[1]
        if not annotation_path.exists():
            raise DeepPCBError(f"{path}:{line_no}: annotation file missing: {annotation_path}")
        pairs.append((image_path, annotation_path))
    return pairs


def _resolve_image_path(pcb: Path, raw_image_path: str, split_path: Path, line_no: int) -> Path:
    image_path = pcb / raw_image_path
    if image_path.exists():
        return image_path

    # DeepPCB split files omit the _test suffix even though files are named *_test.jpg.
    suffixed = image_path.with_name(f"{image_path.stem}_test{image_path.suffix}")
    if suffixed.exists():
        return suffixed

    raise DeepPCBError(f"{split_path}:{line_no}: image file missing: {image_path} or {suffixed}")


def parse_annotation_file(path: Path, image_size: tuple[int, int]) -> tuple[Annotation, ...]:
    width, height = image_size
    annotations: list[Annotation] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise DeepPCBError(f"{path}:{line_no}: expected 5 fields x1 y1 x2 y2 class_id, got {len(parts)}")
        try:
            x1, y1, x2, y2, class_id = (int(value) for value in parts)
        except ValueError as exc:
            raise DeepPCBError(f"{path}:{line_no}: annotation fields must be integers: {raw!r}") from exc
        if class_id not in CLASS_ID_TO_NAME:
            raise DeepPCBError(f"{path}:{line_no}: unknown class id {class_id}")
        if x2 <= x1 or y2 <= y1:
            raise DeepPCBError(f"{path}:{line_no}: zero-area or negative bbox: {raw!r}")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise DeepPCBError(f"{path}:{line_no}: bbox out of bounds for {width}x{height}: {raw!r}")
        annotations.append(Annotation(float(x1), float(y1), float(x2), float(y2), class_id))
    if not annotations:
        raise DeepPCBError(f"{path}: no annotations found")
    return tuple(annotations)


def load_samples(data_root: str | Path | None = None, split_file: str = "trainval.txt") -> list[DeepPCBSample]:
    samples: list[DeepPCBSample] = []
    for image_path, annotation_path in read_split_pairs(data_root, split_file):
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = parse_annotation_file(annotation_path, (width, height))
        samples.append(DeepPCBSample(image_path, annotation_path, annotations, width, height))
    return samples


def train_val_test_samples(
    data_root: str | Path | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[list[DeepPCBSample], list[DeepPCBSample], list[DeepPCBSample]]:
    if not 0.0 < val_fraction < 1.0:
        raise DeepPCBError("val_fraction must be between 0 and 1")
    trainval = load_samples(data_root, "trainval.txt")
    test = load_samples(data_root, "test.txt")
    rng = Random(seed)
    shuffled = list(trainval)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val = shuffled[:val_count]
    train = shuffled[val_count:]
    if max_samples is not None:
        train = train[:max_samples]
        val = val[: max(1, min(len(val), max_samples))]
        test = test[: max(1, min(len(test), max_samples))]
    return train, val, test


def draw_annotated_sample(sample: DeepPCBSample, output_path: Path) -> None:
    image = Image.open(sample.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for ann in sample.annotations:
        draw.rectangle([ann.x1, ann.y1, ann.x2, ann.y2], outline=(220, 38, 38), width=2)
        draw.text((ann.x1, max(0, ann.y1 - 12)), CLASS_ID_TO_NAME[ann.class_id], fill=(220, 38, 38))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def class_distribution(samples: Iterable[DeepPCBSample]) -> dict[int, int]:
    counts = {class_id: 0 for class_id in CLASS_ID_TO_NAME}
    for sample in samples:
        for ann in sample.annotations:
            counts[ann.class_id] += 1
    return counts
