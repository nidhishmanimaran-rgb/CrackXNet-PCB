from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from crackxnet_app.config import DEFAULT_DATA_ROOT, HybridDetectorConfig
from crackxnet_app.deeppcb.dataset import train_val_test_samples
from crackxnet_app.deeppcb.evaluation import full_metrics
from crackxnet_app.deeppcb.model import (
    create_faster_rcnn,
    get_device,
    load_checkpoint,
    runtime_environment,
    save_checkpoint,
    seed_worker,
    set_reproducible_seed,
)
from crackxnet_app.deeppcb.torch_dataset import DeepPCBTorchDataset
from crackxnet_app.deeppcb.transforms import DeepPCBTransforms, TransformConfig, collate_fn
from crackxnet_app.models.crackxnet_detector import create_hybrid_faster_rcnn, load_hybrid_checkpoint, save_hybrid_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DeepPCB Faster R-CNN baseline or CrackXNet hybrid detector.")
    parser.add_argument("--mode", choices=["baseline", "hybrid"], default="baseline")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "trained" / "baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, help="Limit train/val/test samples for smoke tests.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from COCO weights.")
    parser.add_argument("--fusion-dim", type=int, default=64)
    parser.add_argument("--fpn-channels", type=int, default=64)
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, _ = train_val_test_samples(args.data_root, args.val_fraction, args.seed, args.max_samples)
    device = get_device(args.device)
    transforms_train = DeepPCBTransforms(TransformConfig(image_size=args.image_size), train=True)
    transforms_eval = DeepPCBTransforms(TransformConfig(image_size=args.image_size), train=False)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        DeepPCBTorchDataset(train_samples, transforms_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        DeepPCBTorchDataset(val_samples, transforms_eval),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    hybrid_config = HybridDetectorConfig(
        enabled=args.mode == "hybrid",
        checkpoint_path=args.output / "best.pth",
        image_size=args.image_size,
        fusion_dim=args.fusion_dim,
        fpn_out_channels=args.fpn_channels,
        local_input_size=args.image_size,
        vit_input_size=args.image_size,
        vit_variant="vit_b_16",
        local_pretrained=not args.no_pretrained,
        vit_pretrained=not args.no_pretrained,
        confidence_threshold=args.confidence_threshold,
        device=str(device),
    )
    if args.mode == "hybrid":
        model = create_hybrid_faster_rcnn(hybrid_config).to(device)
    else:
        model = create_faster_rcnn(pretrained=not args.no_pretrained, image_size=args.image_size).to(device)
    start_epoch = 0
    best_f1 = -1.0
    optimizer_state = None
    if args.resume:
        if args.mode == "hybrid":
            model, payload = load_hybrid_checkpoint(args.resume, str(device))
            model.to(device)
            checkpoint_image_size = int(payload["hybrid_config"].get("image_size", args.image_size))
        else:
            model, payload = load_checkpoint(args.resume, device, pretrained=False)
            model.to(device)
            checkpoint_image_size = int(payload.get("image_size", args.image_size))
        if checkpoint_image_size != args.image_size:
            raise SystemExit(
                f"Resume checkpoint image size is {checkpoint_image_size}, but --image-size is {args.image_size}. "
                "Use the checkpoint image size or start a new run."
            )
        if "optimizer_state" in payload:
            optimizer_state = payload["optimizer_state"]
        start_epoch = int(payload.get("epoch", 0))
        best_f1 = float(payload.get("metrics", {}).get("f1", -1.0))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=0.0005)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    training_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    training_config["runtime_environment"] = runtime_environment()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        batches = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(value for value in losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1

        metrics = _evaluate_loader(model, val_loader, device, args.confidence_threshold)
        avg_loss = total_loss / max(1, batches)
        checkpoint = args.output / f"epoch_{epoch:03d}.pth"
        if args.mode == "hybrid":
            save_hybrid_checkpoint(
                checkpoint, model, optimizer, epoch, {"loss": avg_loss, **metrics}, hybrid_config, training_config
            )
        else:
            save_checkpoint(
                checkpoint, model, optimizer, epoch, {"loss": avg_loss, **metrics}, image_size=args.image_size,
                training_config=training_config,
            )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            if args.mode == "hybrid":
                save_hybrid_checkpoint(
                    args.output / "best.pth", model, optimizer, epoch, {"loss": avg_loss, **metrics}, hybrid_config,
                    training_config,
                )
            else:
                save_checkpoint(
                    args.output / "best.pth", model, optimizer, epoch, {"loss": avg_loss, **metrics}, image_size=args.image_size,
                    training_config=training_config,
                )
        elapsed = time.time() - started
        print(
            f"epoch={epoch} loss={avg_loss:.6f} precision={metrics['precision']:.6f} "
            f"recall={metrics['recall']:.6f} f1={metrics['f1']:.6f} checkpoint={checkpoint} elapsed={elapsed:.1f}s"
        )


def _evaluate_loader(model, loader, device, threshold: float) -> dict:
    model.eval()
    predictions = []
    targets_all = []
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
    return full_metrics(predictions, targets_all, precision_recall_confidence_threshold=threshold)


if __name__ == "__main__":
    main()
