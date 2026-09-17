# CrackXNet PCB Defect Inspection

Workflow:

```text
PCB image -> detection -> class/severity -> explainability heatmap -> PASS/REWORK/REJECT -> report
```

Phase 2 implements a real DeepPCB + Faster R-CNN baseline. Phase 3 adds the EfficientNet-B0 + CBAM local feature extraction branch. Phase 4 adds the Vision Transformer global feature branch. It preserves the FastAPI UI, CLI inference, reports, and the heuristic detector as explicit demo/fallback mode. It does not implement DDAFF, and it does not claim reproduction of the paper's mAP.

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
  features/
    cbam.py
    ddaff.py
    efficientnet_cbam.py
    vit.py
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

## Phase 3 Local Features

EfficientNet-B0 extracts fine-grained local PCB features from an image tensor. The classification head is removed and the spatial feature map is preserved. CBAM then applies channel attention followed by spatial attention while keeping the tensor shape unchanged.

Module locations:

```text
crackxnet_app/features/cbam.py
crackxnet_app/features/efficientnet_cbam.py
```

Feature flow:

```text
NCHW image tensor
  -> EfficientNet-B0 features
  -> CBAM channel + spatial attention
  -> LocalFeatureOutput(feature_map, pooled_features)
```

For an input tensor shaped `(1, 3, 224, 224)`, the EfficientNet-B0 + CBAM feature map has 1280 channels and retains spatial dimensions from the backbone output. The pooled feature vector has shape `(1, 1280)`.

Configuration is centralized through `LocalFeatureConfig` in `crackxnet_app/config.py`:

- `CRACKXNET_EFFICIENTNET_PRETRAINED`
- `CRACKXNET_LOCAL_FEATURE_SIZE`
- `CRACKXNET_CBAM_REDUCTION`
- `CRACKXNET_LOCAL_FEATURE_DEVICE`
- `CRACKXNET_LOCAL_FEATURE_CHECKPOINT`

`pretrained=False` passes `weights=None` to torchvision and does not request pretrained weights. ViT and DDAFF are not implemented in Phase 3.

Run the feature tests:

```bash
python -m pytest tests\test_phase3_local_features.py
```

## Phase 4 Global Features

The ViT branch extracts global PCB context independently from the Faster R-CNN detector and the EfficientNet-B0 + CBAM local branch. It is intentionally modular so Phase 5 can fuse local and global representations with DDAFF.

Module location:

```text
crackxnet_app/features/vit.py
```

Feature flow:

```text
NCHW image tensor
  -> ViT patch embedding
  -> transformer encoder
  -> GlobalFeatureOutput(class_token, patch_tokens, pooled_features)
```

For `vit_b_16` with `(batch, 3, 224, 224)` input:

- `class_token`: `(batch, 768)`
- `patch_tokens`: `(batch, 196, 768)`
- `pooled_features`: `(batch, 768)`

For smoke tests using `input_size=64`, `vit_b_16` returns `16` patch tokens.

Configuration is centralized through `GlobalFeatureConfig` in `crackxnet_app/config.py`:

- `CRACKXNET_VIT_VARIANT`
- `CRACKXNET_VIT_PRETRAINED`
- `CRACKXNET_VIT_INPUT_SIZE`
- `CRACKXNET_VIT_PATCH_SIZE`
- `CRACKXNET_VIT_DEVICE`
- `CRACKXNET_VIT_CHECKPOINT`

Supported variants are `vit_b_16` and `vit_b_32`. `pretrained=False` passes `weights=None` to torchvision and does not request pretrained weights. Phase 4 adds ViT only; DDAFF remains reserved for Phase 5.

Run the ViT feature tests:

```bash
python -m pytest tests\test_phase4_vit_features.py
```

## Phase 5 Feature Fusion

DDAFF fuses the local EfficientNet-B0 + CBAM stream with the global ViT stream. It is a feature module only; Phase 6 will connect fused features to a detector.

Architecture:

```text
EfficientNet-B0 + CBAM
        |
   Local Features
        \
         -> DDAFF -> Fused Features
        /
      ViT
        |
  Global Features
```

DDAFF projects both streams to a shared fusion dimension, reshapes ViT patch tokens into a spatial grid, aligns them to the local feature map size, computes adaptive trainable local/global stream weights, and returns a detector-ready fused spatial representation.

Current tensor flow:

```text
local feature map:      (batch, local_channels, height, width)
global patch tokens:    (batch, num_patches, global_dim)
local projected:        (batch, fusion_dim, height, width)
global projected:       (batch, fusion_dim, height, width)
fusion weights:         (batch, 2)
fused feature map:      (batch, fusion_dim, height, width)
pooled fused features:  (batch, fusion_dim)
```

Configuration is centralized through `DDAFFConfig` in `crackxnet_app/config.py`:

- `CRACKXNET_DDAFF_FUSION_DIM`
- `CRACKXNET_DDAFF_DROPOUT`
- `CRACKXNET_DDAFF_LAYER_NORM`
- `CRACKXNET_DDAFF_DEVICE`

Run the DDAFF tests:

```bash
python -m pytest tests\test_phase5_ddaff.py -q
```

Smoke-test flow:

```text
real PCB image -> EfficientNet-B0 + CBAM -> ViT -> DDAFF
```

Phase 5 implements feature fusion only. DDAFF is not connected to the production Faster R-CNN detector yet.

## Troubleshooting

- If `D:\PCB\data\DeepPCB` is missing, place or link the local DeepPCB dataset there.
- CPU is supported but slow; use `--device cuda` only when CUDA is available.
- If a checkpoint is missing or invalid, CLI inference exits with a clear error.
- Dataset and model weights are ignored by `.gitignore`; do not commit DeepPCB data or `.pth` files.
