from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from crackxnet_app.config import DEFAULT_DATA_ROOT
from crackxnet_app.deeppcb.dataset import train_val_test_samples
from crackxnet_app.deeppcb.evaluation import full_metrics, save_metrics
from crackxnet_app.deeppcb.model import get_device, load_checkpoint, runtime_environment
from crackxnet_app.deeppcb.torch_dataset import DeepPCBTorchDataset
from crackxnet_app.deeppcb.transforms import DeepPCBTransforms, TransformConfig, collate_fn
from crackxnet_app.models.crackxnet_detector import load_hybrid_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DeepPCB Faster R-CNN baseline or CrackXNet hybrid checkpoint.")
    parser.add_argument("--mode", choices=["baseline", "hybrid"], default="baseline")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640, help="Baseline evaluation transform size; hybrid uses its checkpoint size.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Threshold for reported precision/recall/F1; AP metrics use all detector scores.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    device = get_device(args.device)
    if args.mode == "hybrid":
        model, payload = load_hybrid_checkpoint(args.model, str(device))
        image_size = int(payload["hybrid_config"]["image_size"])
    else:
        model, payload = load_checkpoint(args.model, device, pretrained=False)
        image_size = int(payload.get("image_size", args.image_size))
    _, val_samples, test_samples = train_val_test_samples(args.data_root, max_samples=args.max_samples)
    samples = val_samples if args.split == "val" else test_samples
    loader = DataLoader(
        DeepPCBTorchDataset(samples, DeepPCBTransforms(TransformConfig(image_size=image_size), train=False)),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    predictions = []
    targets_all = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                predictions.append(
                    {
                        "boxes": output["boxes"].detach().cpu(),
                        "labels": output["labels"].detach().cpu(),
                        "scores": output["scores"].detach().cpu(),
                    }
                )
                targets_all.append({"boxes": target["boxes"].detach().cpu(), "labels": target["labels"].detach().cpu()})
    metrics = full_metrics(
        predictions,
        targets_all,
        precision_recall_confidence_threshold=args.confidence_threshold,
    )
    metrics["model_mode"] = args.mode
    metrics["checkpoint"] = str(args.model)
    metrics["evaluation_config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metrics["evaluation_config"]["runtime_environment"] = runtime_environment()
    metrics["split"] = args.split
    metrics["sample_count"] = len(samples)
    save_metrics(metrics, args.output)
    print((args.output / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
