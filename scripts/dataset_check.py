from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackxnet_app.config import CLASS_ID_TO_NAME, DEFAULT_DATA_ROOT
from crackxnet_app.deeppcb.dataset import (
    DeepPCBError,
    class_distribution,
    draw_annotated_sample,
    load_samples,
    train_val_test_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize a local DeepPCB dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "dataset_check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=4, help="Number of annotated preview images to write.")
    args = parser.parse_args()

    try:
        train, val, test = train_val_test_samples(args.data_root, args.val_fraction, args.seed)
        all_samples = train + val + test
        dims = Counter((sample.width, sample.height) for sample in all_samples)
        classes = class_distribution(all_samples)
        trainval = load_samples(args.data_root, "trainval.txt")
    except DeepPCBError as exc:
        raise SystemExit(f"Dataset check failed: {exc}") from exc

    args.output.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(all_samples[: args.samples], start=1):
        draw_annotated_sample(sample, args.output / f"annotated_sample_{index}.jpg")

    summary = {
        "data_root": str(args.data_root),
        "total_images": len(all_samples),
        "annotation_count": sum(len(sample.annotations) for sample in all_samples),
        "class_distribution": {CLASS_ID_TO_NAME[k]: v for k, v in classes.items()},
        "train_count": len(train),
        "validation_count": len(val),
        "test_count": len(test),
        "trainval_count": len(trainval),
        "image_dimensions": {f"{w}x{h}": count for (w, h), count in dims.items()},
        "invalid_or_missing_data": 0,
        "bbox_validity": "all parsed boxes are valid",
        "preview_dir": str(args.output.resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
