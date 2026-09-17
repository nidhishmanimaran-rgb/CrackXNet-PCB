const fileInput = document.querySelector("#fileInput");
const decision = document.querySelector("#decision");
const defectCount = document.querySelector("#defectCount");
const maxSeverity = document.querySelector("#maxSeverity");
const overlay = document.querySelector("#overlay");
const heatmap = document.querySelector("#heatmap");
const findings = document.querySelector("#findings");
const message = document.querySelector("#message");
const reportLink = document.querySelector("#reportLink");

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  message.textContent = "Inspecting image...";
  reportLink.classList.add("disabled");

  const form = new FormData();
  form.append("file", file);

  try {
    const response = await fetch("/api/inspect", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Inspection failed.");
    }
    renderResult(payload);
    message.textContent = payload.notes.join(" ");
  } catch (error) {
    message.textContent = error.message;
  }
});

function renderResult(payload) {
  decision.textContent = payload.decision;
  decision.className = `decision-${payload.decision.toLowerCase()}`;
  defectCount.textContent = payload.defects.length;
  maxSeverity.textContent = Number(payload.max_severity).toFixed(3);
  overlay.src = payload.overlay_image;
  heatmap.src = payload.heatmap_image;
  reportLink.href = `/api/report/${payload.report_id}`;
  reportLink.classList.remove("disabled");

  if (!payload.defects.length) {
    findings.innerHTML = '<tr><td colspan="5">No defects detected by the baseline pipeline.</td></tr>';
    return;
  }

  findings.innerHTML = payload.defects.map((defect) => {
    const box = defect.bbox;
    return `<tr>
      <td>${escapeHtml(defect.label)}</td>
      <td>${Number(defect.confidence).toFixed(3)}</td>
      <td>${Number(defect.severity).toFixed(3)}</td>
      <td>${box.x1}, ${box.y1}, ${box.x2}, ${box.y2}</td>
      <td>${escapeHtml(defect.rationale)}</td>
    </tr>`;
  }).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
