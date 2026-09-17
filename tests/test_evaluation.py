from __future__ import annotations

import torch

from crackxnet_app.deeppcb.evaluation import full_metrics


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
