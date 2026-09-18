from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from PIL import Image

from crackxnet_app.api import app
from crackxnet_app.inference.pipeline import CrackXNetPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight CrackXNet latency smoke check.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to time direct real pipeline loading/inference.")
    parser.add_argument("--mode", choices=["baseline", "hybrid"], default="hybrid")
    parser.add_argument("--demo", action="store_true", help="Run explicit demo detector for UI/performance plumbing only.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "audit" / "performance_smoke.json")
    args = parser.parse_args()
    if args.demo and args.checkpoint:
        raise SystemExit("--demo cannot be combined with --checkpoint.")
    if not args.demo and not args.checkpoint:
        raise SystemExit("Real performance smoke requires --checkpoint. Use --demo only for explicit development/demo mode.")

    image = Image.new("RGB", (args.image_size, args.image_size), (42, 96, 64))
    measurements: dict[str, float | str | int | None] = {
        "image_size": args.image_size,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
    }

    started = time.perf_counter()
    pipeline = CrackXNetPipeline(force_demo=args.demo, checkpoint_path=args.checkpoint, model_mode=args.mode, device=args.device)
    measurements["pipeline_init_seconds"] = round(time.perf_counter() - started, 4)

    started = time.perf_counter()
    outputs = pipeline.inspect(image, filename="performance-smoke.png")
    measurements["pipeline_inspect_seconds"] = round(time.perf_counter() - started, 4)
    measurements["decision"] = outputs.result.decision
    measurements["defects"] = len(outputs.result.defects)
    measurements["model_status"] = pipeline.model_status

    payload = io.BytesIO()
    image.save(payload, format="PNG")
    client = TestClient(app)
    started = time.perf_counter()
    response = client.post("/api/inspect", files={"file": ("performance-smoke.png", payload.getvalue(), "image/png")})
    measurements["api_global_inspect_seconds"] = round(time.perf_counter() - started, 4)
    measurements["api_global_status"] = response.status_code
    if response.status_code == 200:
        body = response.json()
        measurements["api_global_decision"] = body.get("decision")
        measurements["api_global_quality_status"] = body.get("quality", {}).get("status")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
