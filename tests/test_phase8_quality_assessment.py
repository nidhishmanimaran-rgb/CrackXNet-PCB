from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from crackxnet_app.api import app
from crackxnet_app.config import QualityConfig
from crackxnet_app.quality import RuleBasedQualityAssessor
from crackxnet_app.reporting.html_report import render_html_report
from crackxnet_app.schemas import BoundingBox, DefectPrediction, InspectionResult


def _defect(label: str = "Pin Hole", confidence: float = 0.9, severity_label: str = "LOW", severity=0.1) -> DefectPrediction:
    return DefectPrediction(
        label=label,
        confidence=confidence,
        severity=severity,
        severity_label=severity_label,
        severity_reason="test severity",
        bbox=BoundingBox(1, 1, 10, 10),
        rationale="test defect",
    )


def _assessor(min_confidence: float = 0.25) -> RuleBasedQualityAssessor:
    return RuleBasedQualityAssessor(
        QualityConfig(
            mode="rule_based",
            min_confidence=min_confidence,
            reject_on_high_severity_count=1,
            reject_on_total_defects=4,
            rework_on_medium_severity_count=1,
            rework_on_low_severity_count=3,
            pass_on_no_detections=True,
        )
    )


def test_quality_pass_zero_detections() -> None:
    result = _assessor().assess([])

    assert result.status == "PASS"
    assert result.defect_count == 0
    assert "No detections" in result.reason


def test_quality_pass_single_low_defect() -> None:
    result = _assessor().assess([_defect(severity_label="LOW")])

    assert result.status == "PASS"
    assert result.low_severity_count == 1


def test_quality_rework_medium_defect() -> None:
    result = _assessor().assess([_defect(severity_label="MEDIUM", severity=0.5)])

    assert result.status == "REWORK"
    assert result.medium_severity_count == 1


def test_quality_reject_high_defect() -> None:
    result = _assessor().assess([_defect(severity_label="HIGH", severity=0.9)])

    assert result.status == "REJECT"
    assert result.high_severity_count == 1


def test_quality_mixed_and_multiple_defects_priority() -> None:
    defects = [
        _defect("Pin Hole", severity_label="LOW"),
        _defect("Mouse Bite", severity_label="MEDIUM", severity=0.5),
        _defect("Open Circuit", severity_label="HIGH", severity=0.9),
    ]
    result = _assessor().assess(defects)

    assert result.status == "REJECT"
    assert result.detected_defect_types == ["Mouse Bite", "Open Circuit", "Pin Hole"]


def test_quality_low_confidence_filtering() -> None:
    result = _assessor(min_confidence=0.5).assess([
        _defect(confidence=0.1, severity_label="HIGH", severity=0.95),
        _defect(confidence=0.9, severity_label="LOW", severity=0.1),
    ])

    assert result.status == "PASS"
    assert result.defect_count == 1
    assert result.ignored_low_confidence_count == 1


def test_quality_deterministic_invalid_missing_severity_unknown_class() -> None:
    weird = _defect("", confidence=0.8, severity_label="", severity="bad")
    weird.bbox = BoundingBox(5, 5, 5, 5)

    first = _assessor().assess([weird])
    second = _assessor().assess([weird])

    assert first == second
    assert first.status == "PASS"
    assert first.low_severity_count == 1
    assert first.detected_defect_types == ["Unknown"]


def test_quality_reject_defect_count() -> None:
    result = _assessor().assess([_defect(label=f"Type {idx}", severity_label="LOW") for idx in range(4)])

    assert result.status == "REJECT"
    assert result.defect_count == 4


def test_api_response_contains_quality(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (96, 96), (30, 100, 60)).save(image_path)
    client = TestClient(app)

    with image_path.open("rb") as handle:
        response = client.post("/api/inspect", files={"file": ("sample.png", handle, "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["quality"] is not None
    assert payload["decision"] == payload["quality"]["status"]
    assert "recommendation" in payload["quality"]
    assert "confidence_summary" in payload["quality"]


def test_report_generation_contains_quality_section() -> None:
    quality = _assessor().assess([_defect(severity_label="MEDIUM", severity=0.5)])
    result = InspectionResult(
        filename="sample.png",
        image_width=100,
        image_height=100,
        decision=quality.status,
        max_severity=0.5,
        defects=[],
        notes=["Model status: test"],
        quality=quality,
    )

    html = render_html_report(result)

    assert "Quality Assessment" in html
    assert "REWORK" in html
    assert "Recommendation" in html
