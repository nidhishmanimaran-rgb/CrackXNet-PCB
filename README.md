# CrackXNet PCB Defect Inspection MVP

This is a runnable MVP for the CrackXNet PCB defect inspection workflow:

`PCB image -> defect detection -> class/severity -> explainability heatmap -> PASS/REWORK/REJECT -> report`

The app is production-shaped but intentionally honest about modeling status. It includes a deterministic baseline detector so the product flow works end to end. It does **not** claim CrackXNet paper accuracy or report fabricated metrics. The EfficientNet-B0+CBAM, ViT, DDAFF, FPN, and Faster R-CNN pieces are separated as extension points for later replacement with trained models.

## Supported Defect Classes

- Open Circuit
- Short Circuit
- Mouse Bite
- Spur
- Pin Hole
- Spurious Copper

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn crackxnet_app.api:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Run Tests

```bash
pytest
```

## CLI Inference

```bash
python scripts/infer.py path\to\pcb.png --out outputs
```

Outputs:

- `inspection.json`
- `overlay.png`
- `heatmap.png`
- `report.html`

## Training Entry Point

```bash
python scripts/train.py --data-root path\to\DeepPCB --epochs 1
```

The current training script validates dataset structure and writes a manifest. It does not train the full CrackXNet hybrid model yet. That is intentional: DeepPCB training, Faster R-CNN fine-tuning, ViT fusion, and DDAFF validation require a real training loop and evaluation budget.

## Architecture

```text
crackxnet_app/
  api.py                 FastAPI app and web endpoints
  config.py              Labels and thresholds
  schemas.py             Typed result models
  inference/
    pipeline.py          End-to-end inspection pipeline
    preprocessing.py     Resize/normalize utilities
    baseline_detector.py Heuristic detector baseline
    modules.py           CrackXNet extension interfaces/stubs
  reporting/
    html_report.py       Downloadable HTML report
  static/
    index.html           Upload UI
    app.js               Browser interaction
    styles.css           Minimal app styling
scripts/
  infer.py               CLI inference
  train.py               Dataset/training scaffold
tests/
  test_pipeline.py       Smoke tests with synthetic PCB image
```

## What Is Real in This MVP

- Upload and inspect a PCB image.
- Produce defect boxes, labels, confidence scores, severity, decision, visual overlay, and heatmap.
- Generate downloadable inspection reports.
- Run CLI inference.
- Run tests.
- Keep paper components modular.

## What Is Baseline or Extension-Ready

- Detection is a classical image-processing baseline, not trained Faster R-CNN.
- Classification uses rule-based geometry/color features, not the final hybrid CrackXNet classifier.
- Severity is rule-based with a clean replacement boundary for learned severity.
- Heatmap is saliency-style baseline from defect masks, not Grad-CAM from a trained neural backbone.
- Metrics are not reported unless you evaluate against labeled data.

## DeepPCB Notes

DeepPCB has six defect classes and paired template/test PCB images with annotations. To implement the full paper model later:

1. Convert annotations into COCO or Pascal VOC format.
2. Fine-tune an FPN Faster R-CNN detector.
3. Add EfficientNet-B0 + CBAM local features.
4. Add ViT global features.
5. Implement DDAFF for adaptive local/global fusion.
6. Evaluate Precision, Recall, F1, mAP@0.5, and mAP@0.5:0.95 on a held-out split.
