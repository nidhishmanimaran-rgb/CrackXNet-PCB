from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from crackxnet_app.config import CLASS_ID_TO_NAME


@dataclass
class MatchStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    if box_a.numel() == 0 or box_b.numel() == 0:
        return torch.zeros((box_a.shape[0], box_b.shape[0]))
    lt = torch.maximum(box_a[:, None, :2], box_b[:, :2])
    rb = torch.minimum(box_a[:, None, 2:], box_b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))[:, None]
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def precision_recall_f1(predictions: list[dict], targets: list[dict], iou_threshold: float = 0.5) -> dict:
    per_class = {class_id: MatchStats() for class_id in CLASS_ID_TO_NAME}
    for pred, target in zip(predictions, targets):
        for class_id in CLASS_ID_TO_NAME:
            pred_mask = pred["labels"] == class_id
            gt_mask = target["labels"] == class_id
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            gt_boxes = target["boxes"][gt_mask]
            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            matched: set[int] = set()
            for box in pred_boxes:
                if len(gt_boxes) == 0:
                    per_class[class_id].fp += 1
                    continue
                ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_idx = torch.max(ious, dim=0)
                if float(best_iou) >= iou_threshold and int(best_idx) not in matched:
                    per_class[class_id].tp += 1
                    matched.add(int(best_idx))
                else:
                    per_class[class_id].fp += 1
            per_class[class_id].fn += max(0, len(gt_boxes) - len(matched))
    return _summarize_stats(per_class)


def average_precision(predictions: list[dict], targets: list[dict], iou_threshold: float) -> dict[int, float]:
    aps: dict[int, float] = {}
    for class_id in CLASS_ID_TO_NAME:
        records = []
        total_gt = 0
        for image_idx, (pred, target) in enumerate(zip(predictions, targets)):
            gt_mask = target["labels"] == class_id
            total_gt += int(gt_mask.sum())
            pred_mask = pred["labels"] == class_id
            for box, score in zip(pred["boxes"][pred_mask], pred["scores"][pred_mask]):
                records.append((float(score), image_idx, box))
        if total_gt == 0:
            aps[class_id] = 0.0
            continue
        records.sort(key=lambda item: item[0], reverse=True)
        matched = {idx: set() for idx in range(len(targets))}
        tp = []
        fp = []
        for _, image_idx, box in records:
            gt_boxes = targets[image_idx]["boxes"][targets[image_idx]["labels"] == class_id]
            if len(gt_boxes) == 0:
                tp.append(0.0)
                fp.append(1.0)
                continue
            ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = torch.max(ious, dim=0)
            if float(best_iou) >= iou_threshold and int(best_idx) not in matched[image_idx]:
                tp.append(1.0)
                fp.append(0.0)
                matched[image_idx].add(int(best_idx))
            else:
                tp.append(0.0)
                fp.append(1.0)
        if not tp:
            aps[class_id] = 0.0
            continue
        tp_t = torch.tensor(tp).cumsum(0)
        fp_t = torch.tensor(fp).cumsum(0)
        recalls = tp_t / max(1, total_gt)
        precisions = tp_t / (tp_t + fp_t).clamp(min=1e-6)
        aps[class_id] = float(_voc_ap(recalls, precisions))
    return aps


def map_metrics(predictions: list[dict], targets: list[dict]) -> dict:
    ap50 = average_precision(predictions, targets, 0.5)
    thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]
    ap_by_threshold = {thr: average_precision(predictions, targets, thr) for thr in thresholds}
    per_class = {}
    for class_id, name in CLASS_ID_TO_NAME.items():
        values = [ap_by_threshold[thr][class_id] for thr in thresholds]
        per_class[name] = {"AP@0.5": ap50[class_id], "AP@0.5:0.95": sum(values) / len(values)}
    return {
        "mAP@0.5": sum(ap50.values()) / len(ap50),
        "mAP@0.5:0.95": sum(v["AP@0.5:0.95"] for v in per_class.values()) / len(per_class),
        "per_class_ap": per_class,
    }


def full_metrics(
    predictions: list[dict],
    targets: list[dict],
    precision_recall_confidence_threshold: float | None = 0.5,
) -> dict:
    """Calculate thresholded precision/recall/F1 and score-ranked AP metrics.

    AP/mAP intentionally use every detector score.  Applying a confidence cutoff
    before AP would make the reported mAP dependent on a deployment threshold.
    """
    pr_predictions = _filter_predictions(predictions, precision_recall_confidence_threshold)
    pr = precision_recall_f1(pr_predictions, targets, iou_threshold=0.5)
    maps = map_metrics(predictions, targets)
    return {
        **pr,
        **maps,
        "precision_recall_confidence_threshold": precision_recall_confidence_threshold,
    }


def _filter_predictions(predictions: list[dict], threshold: float | None) -> list[dict]:
    if threshold is None:
        return predictions
    return [
        {
            "boxes": prediction["boxes"][prediction["scores"] >= threshold],
            "labels": prediction["labels"][prediction["scores"] >= threshold],
            "scores": prediction["scores"][prediction["scores"] >= threshold],
        }
        for prediction in predictions
    ]


def save_metrics(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "AP@0.5", "AP@0.5:0.95"])
        for class_name, values in metrics["per_class"].items():
            ap = metrics["per_class_ap"][class_name]
            writer.writerow([class_name, values["precision"], values["recall"], values["f1"], ap["AP@0.5"], ap["AP@0.5:0.95"]])
    summary = [
        "DeepPCB Faster R-CNN Evaluation",
        "IoU matching: one prediction can match one same-class ground-truth box.",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall: {metrics['recall']:.6f}",
        f"F1-score: {metrics['f1']:.6f}",
        f"mAP@0.5: {metrics['mAP@0.5']:.6f}",
        f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")


def _voc_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    indices = torch.where(mrec[1:] != mrec[:-1])[0]
    return float(torch.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def _summarize_stats(per_class_stats: dict[int, MatchStats]) -> dict:
    per_class = {}
    total = MatchStats()
    for class_id, stats in per_class_stats.items():
        total.tp += stats.tp
        total.fp += stats.fp
        total.fn += stats.fn
        per_class[CLASS_ID_TO_NAME[class_id]] = _stats_to_metrics(stats)
    overall = _stats_to_metrics(total)
    return {"precision": overall["precision"], "recall": overall["recall"], "f1": overall["f1"], "per_class": per_class}


def _stats_to_metrics(stats: MatchStats) -> dict[str, float | int]:
    precision = stats.tp / max(1, stats.tp + stats.fp)
    recall = stats.tp / max(1, stats.tp + stats.fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return {"tp": stats.tp, "fp": stats.fp, "fn": stats.fn, "precision": precision, "recall": recall, "f1": f1}
