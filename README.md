# CrackXNet PCB Defect Inspection

Workflow:

```text
PCB image -> detection -> class/severity -> explainability heatmap -> PASS/REWORK/REJECT -> report
```

Phase 2 implements a real DeepPCB + Faster R-CNN baseline. It preserves the FastAPI UI, CLI inference, reports, and the heuristic detector as explicit demo/fallback mode. It does not implement EfficientNet-B0 + CBAM, ViT, or DDAFF, and it does not claim reproduction of the paper's mAP.

## Classes

DeepPCB class IDs:

- `1`: Open Circuit
- `2`: Short Circuit
- `3`: Mouse Bite
- `4`: Spur
- `5`: Spurious Copper
- `6`: Pin Hole

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Default dataset path:

```text
D:\PCB\data\DeepPCB
```

Detected structure:

```text
DeepPCB/
  PCBData/
    trainval.txt
    test.txt
    group*/
      <group>/
        *_test.jpg
        *_temp.jpg
      <group>_not/
        *.txt
```

Split files contain:

```text
image_path annotation_path
```

The split image paths omit `_test`, while files are named `*_test.jpg`; the loader resolves both forms. Annotation format is:

```text
x1 y1 x2 y2 class_id
```

## Dataset Check

```bash
python scripts\dataset_check.py --data-root D:\PCB\data\DeepPCB
```

Reports image counts, annotation count, class distribution, train/validation/test counts, dimensions, missing/invalid data, and bbox validity. Annotated preview images are written only under `outputs/dataset_check`.

## Train

Full CPU training may be slow:

```bash
python scripts\train.py --data-root D:\PCB\data\DeepPCB --epochs 10 --batch-size 2 --lr 0.005 --output outputs\checkpoints --device auto
```

Smoke test:

```bash
python scripts\train.py --data-root D:\PCB\data\DeepPCB --epochs 1 --batch-size 1 --max-samples 1 --no-pretrained
```

Training logs include epoch, loss, validation metrics, checkpoint path, and elapsed time. Checkpoints are saved under `outputs/checkpoints`, with `best.pth` selected by real validation F1.

## Evaluate

```bash
python scripts\evaluate.py --data-root D:\PCB\data\DeepPCB --model outputs\checkpoints\best.pth --output outputs\evaluation
```

Metrics are computed with documented one-to-one same-class IoU matching:

- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95
- Per-class metrics for all six classes

Outputs: `metrics.json`, `metrics.csv`, and `summary.txt`.

## Inference

Checkpoint-backed inference:

```bash
python scripts\infer.py path\to\pcb.png --model outputs\checkpoints\best.pth --out outputs
```

Explicit demo mode:

```bash
python scripts\infer.py path\to\pcb.png --out outputs
```

Outputs:

- `inspection.json`
- `overlay.png`
- `heatmap.png`
- `report.html`

## FastAPI

```bash
python -m uvicorn crackxnet_app.api:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The app auto-loads `outputs/checkpoints/best.pth` unless overridden:

```bash
set CRACKXNET_CHECKPOINT=D:\PCB\outputs\checkpoints\best.pth
set CRACKXNET_DEVICE=auto
set CRACKXNET_CONFIDENCE=0.5
```

The UI shows `DeepPCB Faster R-CNN` when a trained checkpoint is loaded. With no checkpoint, it shows `Baseline / Demo Mode`.

## Architecture

```text
crackxnet_app/
  api.py
  config.py
  schemas.py
  deeppcb/
    dataset.py
    transforms.py
    torch_dataset.py
    model.py
    evaluation.py
  inference/
    baseline_detector.py
    faster_rcnn_detector.py
    pipeline.py
    preprocessing.py
    modules.py
  reporting/
    html_report.py
  static/
scripts/
  dataset_check.py
  train.py
  evaluate.py
  infer.py
tests/
```

## Phase Roadmap

- Phase 2: DeepPCB + Faster R-CNN baseline.
- Phase 3: EfficientNet-B0 + CBAM local feature module.
- Later phases: ViT global features and DDAFF fusion.

## Troubleshooting

- If `D:\PCB\data\DeepPCB` is missing, place or link the local DeepPCB dataset there.
- CPU is supported but slow; use `--device cuda` only when CUDA is available.
- If a checkpoint is missing or invalid, CLI inference exits with a clear error.
- Dataset and model weights are ignored by `.gitignore`; do not commit DeepPCB data or `.pth` files.
