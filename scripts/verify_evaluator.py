from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from crackxnet_app.deeppcb.evaluation import full_metrics
from crackxnet_app.deeppcb.evaluation_reference import independent_full_metrics


def main() -> None:
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
    tolerance = 1e-6
    metrics = ("precision", "recall", "f1", "mAP@0.5", "mAP@0.5:0.95")
    comparison = {
        name: {
            "project": project[name],
            "independent": independent[name],
            "difference": abs(project[name] - independent[name]),
            "pass": abs(project[name] - independent[name]) <= tolerance,
        }
        for name in metrics
    }
    payload = {"reference": "independent pure-Python IoU/matching/AP implementation", "tolerance": tolerance, "comparison": comparison, "passed": all(item["pass"] for item in comparison.values())}
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit("Independent evaluator verification failed.")


if __name__ == "__main__":
    main()
