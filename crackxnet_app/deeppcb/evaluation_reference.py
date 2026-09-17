"""Independent, pure-Python metric reference used only for evaluator audits.

This module deliberately does not import or call the production evaluator.
It exists to cross-check matching and AP aggregation on deterministic fixtures.
"""
from __future__ import annotations

from collections import defaultdict

from crackxnet_app.config import CLASS_ID_TO_NAME


def independent_full_metrics(predictions: list[dict], targets: list[dict], confidence_threshold: float = 0.5) -> dict[str, float]:
    normalized_predictions = [_prediction_rows(item) for item in predictions]
    normalized_targets = [_target_rows(item) for item in targets]
    tp = fp = fn = 0
    for class_id in CLASS_ID_TO_NAME:
        for prediction, target in zip(normalized_predictions, normalized_targets):
            class_predictions = sorted((row for row in prediction if row[1] == class_id and row[2] >= confidence_threshold), reverse=True)
            class_targets = [row[0] for row in target if row[1] == class_id]
            matched: set[int] = set()
            for box, _label, _score in class_predictions:
                index, iou = _best_match(box, class_targets)
                if index is not None and iou >= 0.5 and index not in matched:
                    tp += 1
                    matched.add(index)
                else:
                    fp += 1
            fn += len(class_targets) - len(matched)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    ap50 = [_average_precision(normalized_predictions, normalized_targets, class_id, 0.5) for class_id in CLASS_ID_TO_NAME]
    map5095 = [
        _average_precision(normalized_predictions, normalized_targets, class_id, threshold)
        for class_id in CLASS_ID_TO_NAME
        for threshold in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    ]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP@0.5": sum(ap50) / len(ap50),
        "mAP@0.5:0.95": sum(map5095) / len(map5095),
    }


def _average_precision(predictions: list[list[tuple]], targets: list[list[tuple]], class_id: int, threshold: float) -> float:
    records = []
    total_targets = 0
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        total_targets += sum(label == class_id for _box, label in target)
        records.extend((score, image_index, box) for box, label, score in prediction if label == class_id)
    if total_targets == 0:
        return 0.0
    records.sort(reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    tp, fp = [], []
    for _score, image_index, box in records:
        target_boxes = [candidate for candidate, label in targets[image_index] if label == class_id]
        index, iou = _best_match(box, target_boxes)
        if index is not None and iou >= threshold and index not in matched[image_index]:
            matched[image_index].add(index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    cumulative_tp, cumulative_fp = 0.0, 0.0
    recalls, precisions = [], []
    for true_positive, false_positive in zip(tp, fp):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        recalls.append(cumulative_tp / total_targets)
        precisions.append(cumulative_tp / max(1e-12, cumulative_tp + cumulative_fp))
    recalls = [0.0, *recalls, 1.0]
    precisions = [0.0, *precisions, 0.0]
    for index in range(len(precisions) - 1, 0, -1):
        precisions[index - 1] = max(precisions[index - 1], precisions[index])
    return sum((recalls[index + 1] - recalls[index]) * precisions[index + 1] for index in range(len(recalls) - 1) if recalls[index + 1] != recalls[index])


def _prediction_rows(prediction: dict) -> list[tuple[tuple[float, float, float, float], int, float]]:
    return [
        (tuple(float(value) for value in box), int(label), float(score))
        for box, label, score in zip(prediction["boxes"].tolist(), prediction["labels"].tolist(), prediction["scores"].tolist())
    ]


def _target_rows(target: dict) -> list[tuple[tuple[float, float, float, float], int]]:
    return [(tuple(float(value) for value in box), int(label)) for box, label in zip(target["boxes"].tolist(), target["labels"].tolist())]


def _best_match(box: tuple[float, float, float, float], candidates: list[tuple[float, float, float, float]]) -> tuple[int | None, float]:
    if not candidates:
        return None, 0.0
    best_index, best_iou = 0, -1.0
    for index, candidate in enumerate(candidates):
        score = _iou(box, candidate)
        if score > best_iou:
            best_index, best_iou = index, score
    return best_index, best_iou


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-12, left_area + right_area - intersection)
