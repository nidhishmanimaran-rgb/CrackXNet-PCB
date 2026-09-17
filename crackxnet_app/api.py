from __future__ import annotations

import base64
import io
import secrets
import time
from collections import OrderedDict
from collections import defaultdict, deque
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from crackxnet_app.inference.pipeline import CrackXNetPipeline
from crackxnet_app.inference.preprocessing import load_rgb_image, resize_for_inference
from crackxnet_app.reporting.html_report import render_html_report
from crackxnet_app.config import DEFAULT_SECURITY_CONFIG, MAX_UPLOAD_BYTES, REPORT_CACHE_MAX_ENTRIES
from crackxnet_app.inference.pipeline import DetectorUnavailableError


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="CrackXNet PCB Inspection MVP", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline = CrackXNetPipeline()
_REPORT_CACHE: OrderedDict[str, str] = OrderedDict()
_REQUEST_TIMESTAMPS: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        if not _is_authorized(request):
            return _with_security_headers(
                JSONResponse({"detail": "Unauthorized."}, status_code=401, headers={"WWW-Authenticate": "ApiKey"})
            )
        if not _within_rate_limit(request):
            return _with_security_headers(
                JSONResponse({"detail": "Rate limit exceeded."}, status_code=429, headers={"Retry-After": str(DEFAULT_SECURITY_CONFIG.rate_limit_window_seconds)})
            )
    response = await call_next(request)
    return _with_security_headers(response)


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
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the configured upload limit.")

    try:
        image = resize_for_inference(load_rgb_image(data))
        outputs = pipeline.inspect(image, filename=_safe_display_filename(file.filename))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DetectorUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Configured detector is unavailable.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inspection failed.") from exc

    payload = outputs.result.to_dict()
    payload["overlay_image"] = _image_to_data_uri(outputs.overlay)
    payload["heatmap_image"] = _image_to_data_uri(outputs.heatmap)
    report_id = str(uuid4())
    report_html = render_html_report(outputs.result, explainability_image_uri=payload["heatmap_image"])
    _cache_report(report_id, report_html)
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


def _cache_report(report_id: str, report_html: str) -> None:
    _REPORT_CACHE[report_id] = report_html
    _REPORT_CACHE.move_to_end(report_id)
    while len(_REPORT_CACHE) > REPORT_CACHE_MAX_ENTRIES:
        _REPORT_CACHE.popitem(last=False)


def _with_security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'self'; frame-ancestors 'none'"
    return response


def _safe_display_filename(filename: str | None) -> str:
    if not filename:
        return "uploaded-image"
    cleaned = Path(filename.replace("\\", "/")).name.strip()
    return cleaned or "uploaded-image"


def _is_authorized(request: Request) -> bool:
    expected = DEFAULT_SECURITY_CONFIG.api_key
    if not expected:
        return True
    received = request.headers.get("X-API-Key", "")
    return secrets.compare_digest(received, expected)


def _within_rate_limit(request: Request) -> bool:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    timestamps = _REQUEST_TIMESTAMPS[client]
    cutoff = now - DEFAULT_SECURITY_CONFIG.rate_limit_window_seconds
    while timestamps and timestamps[0] <= cutoff:
        timestamps.popleft()
    if len(timestamps) >= DEFAULT_SECURITY_CONFIG.rate_limit_requests:
        return False
    timestamps.append(now)
    return True
