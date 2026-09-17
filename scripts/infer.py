from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackxnet_app.inference.pipeline import CrackXNetPipeline
from crackxnet_app.inference.preprocessing import resize_for_inference
from crackxnet_app.reporting.html_report import render_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrackXNet MVP inference on one PCB image.")
    parser.add_argument("image", type=Path, help="Input PCB image path.")
    parser.add_argument("--model", type=Path, help="Faster R-CNN checkpoint path. If omitted, explicit demo mode is used.")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Input image does not exist: {args.image}")

    args.out.mkdir(parents=True, exist_ok=True)
    image = resize_for_inference(Image.open(args.image).convert("RGB"))
    if args.model and not args.model.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.model}")
    pipeline = CrackXNetPipeline(
        checkpoint_path=args.model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        force_demo=args.model is None,
    )
    outputs = pipeline.inspect(image, filename=args.image.name)

    (args.out / "inspection.json").write_text(json.dumps(outputs.result.to_dict(), indent=2), encoding="utf-8")
    (args.out / "report.html").write_text(render_html_report(outputs.result), encoding="utf-8")
    outputs.overlay.save(args.out / "overlay.png")
    outputs.heatmap.save(args.out / "heatmap.png")

    print(f"Decision: {outputs.result.decision}")
    print(f"Defects: {len(outputs.result.defects)}")
    print(f"Model: {pipeline.model_status}")
    print(f"Outputs written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
