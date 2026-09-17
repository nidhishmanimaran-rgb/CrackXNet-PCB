from __future__ import annotations

from collections import Counter
from statistics import mean

from crackxnet_app.config import DEFAULT_QUALITY_CONFIG, QualityConfig
from crackxnet_app.schemas import DefectPrediction, QualityAssessment


class RuleBasedQualityAssessor:
    """Transparent PASS/REWORK/REJECT layer.

    This is not a trained quality classifier. It applies deterministic,
    configurable engineering rules to severity-enriched detections.
    Priority order:
    1. REJECT for high-severity defects or excessive defect count.
    2. REWORK for medium defects or repeated low-severity defects.
    3. PASS when no configured quality issue is present.
    """

    VALID_STATUSES = {"PASS", "REWORK", "REJECT"}
    VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}

    def __init__(self, config: QualityConfig = DEFAULT_QUALITY_CONFIG) -> None:
        if config.mode != "rule_based":
            raise ValueError("Only rule_based quality mode is implemented.")
        if not 0.0 <= config.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        for name in (
            "reject_on_high_severity_count",
            "reject_on_total_defects",
            "rework_on_medium_severity_count",
            "rework_on_low_severity_count",
        ):
            if getattr(config, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        self.config = config

    def assess(self, defects: list[DefectPrediction] | None) -> QualityAssessment:
        defects = list(defects or [])
        considered = [defect for defect in defects if self._confidence(defect) >= self.config.min_confidence]
        ignored_count = len(defects) - len(considered)
        severity_counts = Counter(self._severity_label(defect) for defect in considered)
        defect_types = sorted({str(getattr(defect, "label", "Unknown") or "Unknown") for defect in considered})
        confidence_summary = self._confidence_summary(considered)

        criteria = [
            f"mode={self.config.mode}",
            f"min_confidence={self.config.min_confidence:.3f}",
            f"reject_on_high_severity_count>={self.config.reject_on_high_severity_count}",
            f"reject_on_total_defects>={self.config.reject_on_total_defects}",
            f"rework_on_medium_severity_count>={self.config.rework_on_medium_severity_count}",
            f"rework_on_low_severity_count>={self.config.rework_on_low_severity_count}",
        ]

        high = severity_counts["HIGH"]
        medium = severity_counts["MEDIUM"]
        low = severity_counts["LOW"]
        count = len(considered)

        if count == 0:
            status = "PASS" if self.config.pass_on_no_detections else "REWORK"
            reason = "No detections met the quality-decision confidence threshold."
            recommendation = "Release board if upstream inspection setup is valid." if status == "PASS" else "Review inspection threshold/settings."
        elif self.config.reject_on_high_severity_count and high >= self.config.reject_on_high_severity_count:
            status = "REJECT"
            reason = f"{high} high-severity defect(s) met or exceeded the reject rule."
            recommendation = "Reject board and route for failure analysis."
        elif self.config.reject_on_total_defects and count >= self.config.reject_on_total_defects:
            status = "REJECT"
            reason = f"{count} considered defect(s) met or exceeded the reject defect-count rule."
            recommendation = "Reject board due to excessive detected defects."
        elif self.config.rework_on_medium_severity_count and medium >= self.config.rework_on_medium_severity_count:
            status = "REWORK"
            reason = f"{medium} medium-severity defect(s) require corrective action."
            recommendation = "Route board for rework and reinspect after correction."
        elif self.config.rework_on_low_severity_count and low >= self.config.rework_on_low_severity_count:
            status = "REWORK"
            reason = f"{low} low-severity defect(s) met or exceeded the rework accumulation rule."
            recommendation = "Review board for minor rework before release."
        else:
            status = "PASS"
            reason = "No configured reject or rework quality rule was triggered."
            recommendation = "Release board subject to normal process controls."

        return QualityAssessment(
            status=status,
            reason=reason,
            defect_count=count,
            high_severity_count=high,
            medium_severity_count=medium,
            low_severity_count=low,
            detected_defect_types=defect_types,
            confidence_summary=confidence_summary,
            recommendation=recommendation,
            criteria=criteria,
            ignored_low_confidence_count=ignored_count,
            mode=self.config.mode,
        )

    def _severity_label(self, defect: DefectPrediction) -> str:
        label = str(getattr(defect, "severity_label", "") or "").upper()
        if label in self.VALID_SEVERITIES:
            return label
        score = self._severity_score(defect)
        if score >= 0.72:
            return "HIGH"
        if score >= 0.35:
            return "MEDIUM"
        return "LOW"

    def _severity_score(self, defect: DefectPrediction) -> float:
        try:
            return max(0.0, min(1.0, float(getattr(defect, "severity", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def _confidence(self, defect: DefectPrediction) -> float:
        try:
            return max(0.0, min(1.0, float(getattr(defect, "confidence", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def _confidence_summary(self, defects: list[DefectPrediction]) -> dict[str, float | int | None]:
        if not defects:
            return {"min": None, "max": None, "mean": None, "count": 0}
        values = [self._confidence(defect) for defect in defects]
        return {
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "mean": round(mean(values), 3),
            "count": len(values),
        }
