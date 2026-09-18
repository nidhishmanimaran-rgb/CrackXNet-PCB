const fileInput = document.querySelector("#fileInput");
const decision = document.querySelector("#decision");
const defectCount = document.querySelector("#defectCount");
const maxSeverity = document.querySelector("#maxSeverity");
const modelStatus = document.querySelector("#modelStatus");
const overlay = document.querySelector("#overlay");
const heatmap = document.querySelector("#heatmap");
const findings = document.querySelector("#findings");
const message = document.querySelector("#message");
const reportLink = document.querySelector("#reportLink");
const qualityStatus = document.querySelector("#qualityStatus");
const qualityDefects = document.querySelector("#qualityDefects");
const qualitySeverity = document.querySelector("#qualitySeverity");
const qualityReason = document.querySelector("#qualityReason");
const qualityRecommendation = document.querySelector("#qualityRecommendation");

loadModelStatus();

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
    overlay.removeAttribute("src");
    heatmap.removeAttribute("src");
    findings.innerHTML = '<tr><td colspan="5">No inference result available.</td></tr>';
  }
});

async function loadModelStatus() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    modelStatus.textContent = payload.model_status || "Model status unavailable";
    if (payload.model_available === false) {
      message.textContent = "Model unavailable — configure a trained CrackXNet checkpoint.";
    }
  } catch (error) {
    message.textContent = "Unable to read model status.";
  }
}

function renderResult(payload) {
  decision.textContent = payload.decision;
  decision.className = `decision-${payload.decision.toLowerCase()}`;
  defectCount.textContent = payload.defects.length;
  maxSeverity.textContent = Number(payload.max_severity).toFixed(3);
  modelStatus.textContent = payload.model_status || "Model status unavailable";
  overlay.src = payload.overlay_image;
  heatmap.src = payload.heatmap_image;
  reportLink.href = `/api/report/${payload.report_id}`;
  reportLink.classList.remove("disabled");
  renderQuality(payload.quality, payload.decision);

  if (!payload.defects.length) {
    findings.innerHTML = '<tr><td colspan="5">No defects detected by the selected inspection pipeline.</td></tr>';
    return;
  }

  findings.innerHTML = payload.defects.map((defect) => {
    const box = defect.bbox;
    return `<tr>
      <td>${escapeHtml(defect.label)}</td>
      <td>${Number(defect.confidence).toFixed(3)}</td>
      <td>${escapeHtml(defect.severity_label || "LOW")} (${Number(defect.severity).toFixed(3)})</td>
      <td>${box.x1}, ${box.y1}, ${box.x2}, ${box.y2}</td>
      <td>${escapeHtml(defect.rationale)} ${escapeHtml(defect.severity_reason || "")}</td>
    </tr>`;
  }).join("");
}

function renderQuality(quality, fallbackDecision) {
  if (!quality) {
    qualityStatus.textContent = fallbackDecision || "-";
    qualityDefects.textContent = "0";
    qualitySeverity.textContent = "-";
    qualityReason.textContent = "";
    qualityRecommendation.textContent = "";
    return;
  }
  qualityStatus.textContent = quality.status;
  qualityStatus.className = `decision-${quality.status.toLowerCase()}`;
  qualityDefects.textContent = `${quality.defect_count} considered, ${quality.ignored_low_confidence_count} ignored`;
  qualitySeverity.textContent = `H:${quality.high_severity_count} M:${quality.medium_severity_count} L:${quality.low_severity_count}`;
  qualityReason.textContent = quality.reason;
  qualityRecommendation.textContent = quality.recommendation;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
