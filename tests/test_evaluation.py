from __future__ import annotations

import torch
import pytest

from crackxnet_app.deeppcb.evaluation import full_metrics
from crackxnet_app.deeppcb.evaluation_reference import independent_full_metrics


def test_full_metrics_for_perfect_single_detection() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]

    metrics = full_metrics(predictions, targets)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mAP@0.5"] > 0.0


def test_map_uses_scores_below_precision_recall_threshold() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.4]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]

    metrics = full_metrics(predictions, targets, precision_recall_confidence_threshold=0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["mAP@0.5"] > 0.0


def test_evaluation_rejects_duplicate_and_wrong_class_predictions() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.9, 0.8]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]

    metrics = full_metrics(predictions, targets, precision_recall_confidence_threshold=0.0)

    assert metrics["per_class"]["Open Circuit"]["tp"] == 1
    assert metrics["per_class"]["Short Circuit"]["fp"] == 1
    assert metrics["precision"] == 0.5


def test_production_metrics_match_independent_reference() -> None:
    predictions = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 28.0, 28.0]]), "labels": torch.tensor([1, 2, 3]), "scores": torch.tensor([0.9, 0.8, 0.4])},
        {"boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([3]), "scores": torch.tensor([0.7])},
    ]
    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]), "labels": torch.tensor([1, 3])},
        {"boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([3])},
    ]
    project = full_metrics(predictions, targets, precision_recall_confidence_threshold=0.5)
    independent = independent_full_metrics(predictions, targets, confidence_threshold=0.5)

    for name in ("precision", "recall", "f1", "mAP@0.5", "mAP@0.5:0.95"):
        assert project[name] == pytest.approx(independent[name], abs=1e-6)
