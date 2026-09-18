from __future__ import annotations

import argparse
import base64
import io
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
    parser = argparse.ArgumentParser(description="Run checkpoint-backed CrackXNet inference on one PCB image.")
    parser.add_argument("image", type=Path, help="Input PCB image path.")
    parser.add_argument("--model", type=Path, help="Checkpoint path for real inference.")
    parser.add_argument("--mode", choices=["baseline", "hybrid"], default="hybrid")
    parser.add_argument("--demo", action="store_true", help="Run the explicit heuristic demo detector instead of real inference.")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Input image does not exist: {args.image}")
    if args.demo and args.model:
        raise SystemExit("--demo cannot be combined with --model.")
    if not args.demo and not args.model:
        raise SystemExit("Real inference requires --model. Use --demo only for explicit development/demo mode.")

    args.out.mkdir(parents=True, exist_ok=True)
    image = resize_for_inference(Image.open(args.image).convert("RGB"))
    if args.model and not args.model.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.model}")
    pipeline = CrackXNetPipeline(
        checkpoint_path=args.model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        force_demo=args.demo,
        model_mode=args.mode,
    )
    outputs = pipeline.inspect(image, filename=args.image.name)

    (args.out / "inspection.json").write_text(json.dumps(outputs.result.to_dict(), indent=2), encoding="utf-8")
    (args.out / "report.html").write_text(
        render_html_report(outputs.result, explainability_image_uri=_image_to_data_uri(outputs.heatmap)),
        encoding="utf-8",
    )
    outputs.overlay.save(args.out / "overlay.png")
    outputs.heatmap.save(args.out / "heatmap.png")

    print(f"Decision: {outputs.result.decision}")
    print(f"Defects: {len(outputs.result.defects)}")
    print(f"Model: {pipeline.model_status}")
    print(f"Outputs written to: {args.out.resolve()}")

def _image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


if __name__ == "__main__":
    main()
