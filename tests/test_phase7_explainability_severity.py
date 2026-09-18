from __future__ import annotations

from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from crackxnet_app import api
from crackxnet_app.api import app
from crackxnet_app.config import ExplainabilityConfig, HybridDetectorConfig, SeverityConfig
from crackxnet_app.explainability import GradCAMExplainer
from crackxnet_app.inference.pipeline import CrackXNetPipeline
from crackxnet_app.models.crackxnet_detector import create_hybrid_faster_rcnn
from crackxnet_app.reporting.html_report import render_html_report
from crackxnet_app.schemas import BoundingBox, DefectPrediction, InspectionResult
from crackxnet_app.severity import RuleBasedSeverityEstimator


def _hybrid_config() -> HybridDetectorConfig:
    return HybridDetectorConfig(
        enabled=True,
        checkpoint_path=Path("outputs/phase7_test/best.pth"),
        image_size=64,
        fusion_dim=16,
        fpn_out_channels=16,
        local_input_size=64,
        vit_input_size=64,
        vit_variant="vit_b_16",
        local_pretrained=False,
        vit_pretrained=False,
        confidence_threshold=0.0,
        device="cpu",
    )


def test_gradcam_explainer_real_activation_gradient_flow() -> None:
    model = create_hybrid_faster_rcnn(_hybrid_config())
    explainer = GradCAMExplainer(ExplainabilityConfig(enabled=True, method="grad_cam", top_k=1))
    image = Image.new("RGB", (64, 64), (30, 100, 60))

    result = explainer.explain_model(model, model.backbone.fusion.norm, image, 64, torch.device("cpu"))

    assert result.target_layer == "backbone.fusion.norm"
    assert result.normalized_heatmap.size == image.size
    assert result.overlay.size == image.size
    assert result.raw_heatmap.numel() > 0
    assert torch.isfinite(result.raw_heatmap).all()
    assert 0.0 <= float(result.raw_heatmap.min()) <= float(result.raw_heatmap.max()) <= 1.0


def test_gradcam_no_detection_handling() -> None:
    explainer = GradCAMExplainer()
    image = Image.new("RGB", (32, 32), (0, 0, 0))
    detector = object()

    result = explainer.explain_detector(detector, image, defects=[])

    assert result.normalized_heatmap.size == image.size
    assert result.raw_heatmap.sum() == 0
    assert "No detections" in result.note


def test_rule_based_severity_labels_are_deterministic() -> None:
    estimator = RuleBasedSeverityEstimator(SeverityConfig(mode="rule_based", low_max_score=0.2, medium_max_score=0.6))
    low = DefectPrediction("Pin Hole", 0.1, 0.0, BoundingBox(1, 1, 5, 5), "small")
    high = DefectPrediction("Short Circuit", 0.95, 0.0, BoundingBox(0, 0, 80, 80), "large")

    first = estimator.estimate(low, (200, 200))
    second = estimator.estimate(low, (200, 200))
    high_result = estimator.estimate(high, (100, 100))

    assert first == second
    assert first.label == "LOW"
    assert high_result.label == "HIGH"
    assert high_result.score > first.score


def test_rule_based_severity_invalid_input() -> None:
    estimator = RuleBasedSeverityEstimator()
    defect = DefectPrediction("Pin Hole", 0.5, 0.0, BoundingBox(1, 1, 1, 5), "bad")

    try:
        estimator.estimate(defect, (100, 100))
    except ValueError as exc:
        assert "positive area" in str(exc)
    else:
        raise AssertionError("Expected invalid bbox to fail severity estimation")


def test_pipeline_adds_structured_severity_and_explainability_note() -> None:
    image = Image.new("RGB", (128, 128), (35, 105, 65))
    pipeline = CrackXNetPipeline(force_demo=True)

    outputs = pipeline.inspect(image, filename="demo.png")

    assert outputs.result.notes
    assert any("Severity mode" in note for note in outputs.result.notes)
    assert any("Explainability" in note for note in outputs.result.notes)
    for defect in outputs.result.defects:
        assert defect.severity_label in {"LOW", "MEDIUM", "HIGH"}
        assert defect.severity_reason


def test_api_response_and_report_compatibility(tmp_path: Path) -> None:
    api.pipeline = CrackXNetPipeline(force_demo=True)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (96, 96), (30, 100, 60)).save(image_path)
    client = TestClient(app)
    with image_path.open("rb") as handle:
        response = client.post("/api/inspect", files={"file": ("sample.png", handle, "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert "heatmap_image" in payload
    assert "overlay_image" in payload
    if payload["defects"]:
        assert "severity_label" in payload["defects"][0]
        assert "severity_reason" in payload["defects"][0]

    result = InspectionResult("sample.png", 96, 96, "PASS", 0.0, notes=["Model status: test"])
    html = render_html_report(result, explainability_image_uri="data:image/png;base64,abc")
    assert "Explainability" in html
    assert "data:image/png;base64,abc" in html
