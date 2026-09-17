from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Training scaffold for CrackXNet/DeepPCB.")
    parser.add_argument("--data-root", type=Path, required=True, help="DeepPCB dataset root.")
    parser.add_argument("--epochs", type=int, default=1, help="Reserved for the future training loop.")
    parser.add_argument("--out", type=Path, default=Path("training_runs/baseline_manifest"))
    args = parser.parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"Dataset root does not exist: {args.data_root}")

    images = sorted(path for path in args.data_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    annotations = sorted(path for path in args.data_root.rglob("*.txt"))
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "data_root": str(args.data_root.resolve()),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "epochs_requested": args.epochs,
        "status": "manifest_only",
        "note": (
            "This MVP does not train the full hybrid CrackXNet model. "
            "Use this manifest validation before adding Faster R-CNN, EfficientNet-CBAM, ViT, and DDAFF training."
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
