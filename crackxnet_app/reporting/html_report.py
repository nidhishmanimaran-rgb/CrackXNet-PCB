from __future__ import annotations

from html import escape

from crackxnet_app.schemas import InspectionResult


def render_html_report(result: InspectionResult) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{escape(defect.label)}</td>"
        f"<td>{defect.confidence:.3f}</td>"
        f"<td>{defect.severity:.3f}</td>"
        f"<td>{defect.bbox.x1},{defect.bbox.y1},{defect.bbox.x2},{defect.bbox.y2}</td>"
        f"<td>{escape(defect.rationale)}</td>"
        "</tr>"
        for index, defect in enumerate(result.defects, start=1)
    )
    if not rows:
        rows = '<tr><td colspan="6">No defects detected by the baseline pipeline.</td></tr>'

    notes = "".join(f"<li>{escape(note)}</li>" for note in result.notes)
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
  </style>
</head>
<body>
  <h1>CrackXNet PCB Inspection Report</h1>
  <p class="muted">File: {escape(result.filename)} | Size: {result.image_width} x {result.image_height}</p>
  <p class="decision">Decision: {escape(result.decision)}</p>
  <p>Max severity: {result.max_severity:.3f}</p>
  <h2>Defects</h2>
  <table>
    <thead>
      <tr><th>#</th><th>Class</th><th>Confidence</th><th>Severity</th><th>Box</th><th>Explanation</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Model Notes</h2>
  <ul>{notes}</ul>
</body>
</html>"""
