from __future__ import annotations

from html import escape

from crackxnet_app.schemas import InspectionResult


def render_html_report(result: InspectionResult, explainability_image_uri: str | None = None) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{escape(defect.label)}</td>"
        f"<td>{defect.confidence:.3f}</td>"
        f"<td>{escape(defect.severity_label)} ({defect.severity:.3f})</td>"
        f"<td>{defect.bbox.x1},{defect.bbox.y1},{defect.bbox.x2},{defect.bbox.y2}</td>"
        f"<td>{escape(defect.rationale)}<br>{escape(defect.severity_reason)}</td>"
        "</tr>"
        for index, defect in enumerate(result.defects, start=1)
    )
    if not rows:
        rows = '<tr><td colspan="6">No defects detected by the selected inspection pipeline.</td></tr>'

    notes = "".join(f"<li>{escape(note)}</li>" for note in result.notes)
    quality = result.quality
    if quality:
        confidence = quality.confidence_summary
        quality_section = f"""
  <h2>Quality Assessment</h2>
  <p class="decision">Status: {escape(quality.status)}</p>
  <p>{escape(quality.reason)}</p>
  <p><strong>Recommendation:</strong> {escape(quality.recommendation)}</p>
  <table>
    <tbody>
      <tr><th>Considered defects</th><td>{quality.defect_count}</td></tr>
      <tr><th>Severity summary</th><td>HIGH: {quality.high_severity_count}, MEDIUM: {quality.medium_severity_count}, LOW: {quality.low_severity_count}</td></tr>
      <tr><th>Ignored low-confidence detections</th><td>{quality.ignored_low_confidence_count}</td></tr>
      <tr><th>Defect types</th><td>{escape(", ".join(quality.detected_defect_types) or "None")}</td></tr>
      <tr><th>Confidence</th><td>min={confidence.get("min")}, mean={confidence.get("mean")}, max={confidence.get("max")}</td></tr>
      <tr><th>Rules</th><td>{escape("; ".join(quality.criteria))}</td></tr>
    </tbody>
  </table>"""
    else:
        quality_section = ""
    explainability = (
        f'<h2>Explainability</h2><img class="heatmap" src="{explainability_image_uri}" alt="Explainability heatmap">'
        if explainability_image_uri
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CrackXNet Inspection Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .decision {{ display: inline-block; padding: 8px 12px; border-radius: 4px; background: #172554; color: white; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    .muted {{ color: #64748b; }}
    .heatmap {{ max-width: 100%; border: 1px solid #cbd5e1; }}
  </style>
</head>
<body>
  <h1>CrackXNet PCB Inspection Report</h1>
  <p class="muted">File: {escape(result.filename)} | Size: {result.image_width} x {result.image_height}</p>
  <p class="decision">Decision: {escape(result.decision)}</p>
  <p>Max severity: {result.max_severity:.3f}</p>
  {quality_section}
  <h2>Defects</h2>
  <table>
    <thead>
      <tr><th>#</th><th>Class</th><th>Confidence</th><th>Severity</th><th>Box</th><th>Explanation</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  {explainability}
  <h2>Model Notes</h2>
  <ul>{notes}</ul>
</body>
</html>"""
