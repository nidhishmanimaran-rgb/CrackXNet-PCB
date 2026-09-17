from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from crackxnet_app.inference.pipeline import CrackXNetPipeline
from crackxnet_app.inference.preprocessing import load_rgb_image, resize_for_inference
from crackxnet_app.reporting.html_report import render_html_report


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="CrackXNet PCB Inspection MVP", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline = CrackXNetPipeline()
_REPORT_CACHE: dict[str, str] = {}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is larger than the 15 MB MVP limit.")

    try:
        image = resize_for_inference(load_rgb_image(data))
        outputs = pipeline.inspect(image, filename=file.filename or "uploaded-image")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inspection failed.") from exc

    report_id = str(uuid4())
    report_html = render_html_report(outputs.result)
    _REPORT_CACHE[report_id] = report_html

    payload = outputs.result.to_dict()
    payload["overlay_image"] = _image_to_data_uri(outputs.overlay)
    payload["heatmap_image"] = _image_to_data_uri(outputs.heatmap)
    payload["report_id"] = report_id
    payload["model_status"] = pipeline.model_status
    return JSONResponse(payload)


@app.get("/api/report/{report_id}")
def download_report(report_id: str) -> Response:
    report = _REPORT_CACHE.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found or expired.")
    return Response(
        report,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="inspection-{report_id}.html"'},
    )


def _image_to_data_uri(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _json_dumps(data: dict) -> str:
    return json.dumps(data, indent=2)
