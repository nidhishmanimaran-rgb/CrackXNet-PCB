# CrackXNet PCB Defect Inspection

Workflow:

```text
PCB image -> detection -> class/severity -> explainability heatmap -> PASS/REWORK/REJECT -> report
```

Phase 2 implements a DeepPCB + Faster R-CNN baseline. Phase 3 adds EfficientNet-B0 + CBAM local features. Phase 4 adds ViT global features. Phase 5 adds DDAFF fusion. Phase 6 integrates the hybrid DDAFF + FPN + Faster R-CNN detector path. Phase 7 adds explainability and transparent rule-based severity reasoning. Phase 8 adds deterministic PASS/REWORK/REJECT quality assessment. Production inference does not silently fall back to heuristic detections; demo mode is explicit and for development only. This implementation does not claim reproduction of the paper's mAP.

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

Default dataset path is project-relative:

```text
data/DeepPCB
```

Override it without editing source when your dataset lives elsewhere:

```powershell
$env:DEEPCB_DATA_ROOT = "path\to\DeepPCB"
```

```bash
export DEEPCB_DATA_ROOT=/path/to/DeepPCB
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
python scripts/dataset_check.py --data-root data/DeepPCB
```

Reports image counts, annotation count, class distribution, train/validation/test counts, dimensions, missing/invalid data, and bbox validity. Annotated preview images are written only under `outputs/dataset_check`.

## Train

Full CPU training may be slow:

```bash
python scripts/train.py --mode baseline --data-root data/DeepPCB --epochs 10 --batch-size 2 --lr 0.005 --image-size 640 --output outputs/trained/baseline --device auto
```

Smoke test:

```bash
python scripts/train.py --data-root data/DeepPCB --epochs 1 --batch-size 1 --max-samples 1 --no-pretrained
```

Training logs include epoch, loss, validation metrics, checkpoint path, and elapsed time. Real checkpoints belong under `outputs/trained/baseline` or `outputs/trained/hybrid`; `best.pth` is selected by real validation F1. Smoke checkpoint directories remain separate and are not trained detector results.

Each newly saved checkpoint records the complete CLI configuration, seed, and runtime package/environment summary. The training script seeds Python, NumPy, Torch, CUDA (when available), and DataLoader workers. Pretrained EfficientNet/ViT branches apply ImageNet normalization; `--no-pretrained` keeps smoke/training inputs unnormalized and avoids weight downloads.

Hybrid smoke training:

```bash
python scripts/train.py --mode hybrid --data-root data/DeepPCB --epochs 1 --batch-size 1 --max-samples 1 --image-size 64 --fusion-dim 16 --fpn-channels 16 --no-pretrained --output outputs/hybrid_smoke --device cpu
```

Real hybrid training (CPU-friendly starting configuration; expect it to be slow):

```bash
python scripts/train.py --mode hybrid --data-root data/DeepPCB --epochs 10 --batch-size 1 --lr 0.001 --image-size 224 --no-pretrained --output outputs/trained/hybrid --device auto
```

## Evaluate

```bash
python scripts/evaluate.py --mode baseline --data-root data/DeepPCB --model outputs/trained/baseline/best.pth --output outputs/evaluation/baseline
python scripts/evaluate.py --mode hybrid --data-root data/DeepPCB --model outputs/trained/hybrid/best.pth --output outputs/evaluation/hybrid
```

Metrics are computed with documented one-to-one same-class IoU matching:

- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95
- Per-class metrics for all six classes

Outputs: `metrics.json`, `metrics.csv`, and `summary.txt`.

Independent evaluator cross-check:

```bash
python scripts\verify_evaluator.py
```

This deterministic smoke check compares the project evaluator against an independent pure-Python reference for precision, recall, F1, mAP@0.5, and mAP@0.5:0.95. It is not a detector-performance result.

## Inference

Checkpoint-backed inference:

```bash
python scripts/infer.py path/to/pcb.png --model outputs/trained/baseline/best.pth --mode baseline --out outputs
```

Hybrid checkpoint inference:

```bash
python scripts/infer.py path/to/pcb.png --model deployment/checkpoints/crackxnet_real_epoch10.pth --mode hybrid --out outputs
```

Explicit demo mode, for development/UI plumbing only:

```bash
python scripts/infer.py path/to/pcb.png --demo --out outputs
```

Without `--model`, real CLI inference exits instead of inventing detections.

Outputs:

- `inspection.json`
- `overlay.png`
- `heatmap.png`
- `report.html`

## FastAPI

Before first local use after cloning, install Git LFS and pull the real checkpoint:

```bash
git lfs install
git lfs pull
```

```bash
python -m uvicorn crackxnet_app.api:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The app defaults to the CrackXNet hybrid architecture and real Git LFS checkpoint:

```text
deployment/checkpoints/crackxnet_real_epoch10.pth
```

Override the trained hybrid checkpoint only when needed:

```bash
set CRACKXNET_MODEL_MODE=hybrid
set CRACKXNET_HYBRID_CHECKPOINT=deployment\checkpoints\crackxnet_real_epoch10.pth
set CRACKXNET_HYBRID_DEVICE=auto
set CRACKXNET_HYBRID_CONFIDENCE=0.5
```

Baseline checkpoints remain supported for comparison, but are not the default:

```bash
set CRACKXNET_MODEL_MODE=baseline
set CRACKXNET_CHECKPOINT=outputs\trained\baseline\best.pth
set CRACKXNET_DEVICE=auto
```

When no valid checkpoint is configured, the UI and API report `Model unavailable — configure a trained CrackXNet checkpoint.` No fake defects are produced in this state. `/api/health` exposes model name, mode, result mode, checkpoint status/path, device, and availability.

Only load checkpoints from trusted sources. Checkpoint loading uses PyTorch `weights_only=True`; an invalid configured checkpoint leaves the app available but returns a clear detector-unavailable response instead of silently falling back to demo mode. The local report cache is bounded by `CRACKXNET_REPORT_CACHE_MAX_ENTRIES` (default `100`).

Optional local API hardening:

```bash
set CRACKXNET_API_KEY=change-me
set CRACKXNET_RATE_LIMIT_REQUESTS=60
set CRACKXNET_RATE_LIMIT_WINDOW_SECONDS=60
```

When `CRACKXNET_API_KEY` is set, `/api/*` endpoints except `/api/health` require `X-API-Key`. The built-in limiter is process-local and suitable for local/demo use; public exposure still requires TLS, a real auth boundary, logging, and infrastructure-level rate limiting.

An example TLS reverse-proxy configuration is provided at `deployment/nginx.example.conf`. Treat it as a starting point only; production deployments still need secret management, persistent logs/storage, monitoring, and an infrastructure-level rate limiter.

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
  explainability/
    gradcam.py
  severity/
    rule_based.py
  quality/
    rule_based.py
  models/
    crackxnet_detector.py
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
- Phase 4: ViT global feature module.
- Phase 5: DDAFF feature fusion.
- Phase 6: DDAFF + FPN + Faster R-CNN hybrid detector integration.
- Phase 7: Explainability head + rule-based defect severity reasoning.
- Phase 8: Rule-based PASS/REWORK/REJECT quality assessment.

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

## Phase 6 Hybrid Detector

Phase 6 connects the fused representation to a detector:

```text
EfficientNet-B0 + CBAM
        +
       ViT
        |
      DDAFF
        |
       FPN
        |
  Faster R-CNN
```

The baseline and hybrid detector paths are selectable:

- `baseline`: torchvision Faster R-CNN MobileNetV3 FPN baseline from Phase 2.
- `hybrid`: EfficientNet-B0 + CBAM, ViT, DDAFF, FPN, Faster R-CNN.

Hybrid implementation location:

```text
crackxnet_app/models/crackxnet_detector.py
```

Smoke-test tensor flow with `image_size=64`:

```text
input:          (batch, 3, 64, 64)
local:          (batch, 1280, 2, 2)
ViT tokens:     (batch, 16, 768)
DDAFF fused:    (batch, fusion_dim, 2, 2)
FPN output:     (batch, fpn_channels, 2, 2)
detector output boxes/labels/scores
```

Phase 6 integrates the architecture and checkpoint path. Full-scale hybrid training and real evaluation remain future work.

## Phase 7 Explainability And Severity

Phase 7 extends inspection output:

```text
Detector output
  -> Explainability Head
  -> Defect Reasoning / Severity
  -> report/API/UI fields
```

Explainability uses Grad-CAM-style gradients from actual detector scores. For the hybrid detector, the target layer is:

```text
model.backbone.fusion.norm
```

This layer is immediately after DDAFF fusion and before FPN, so the heatmap reflects the fused local/global representation used by the detector. The explainer sums the top-k detection scores, backpropagates to the target layer, normalizes the heatmap to `[0, 1]`, resizes it to the input image, and returns both heatmap and overlay. If there are no detections above threshold, the explainer returns an empty heatmap and documents that no Grad-CAM target was selected.

Severity is deterministic and rule-based, not learned. The severity head uses:

- defect confidence
- bounding-box area ratio
- bounding-box elongation

Outputs:

- `LOW`
- `MEDIUM`
- `HIGH`

Configuration is centralized through `SeverityConfig` and `ExplainabilityConfig` in `crackxnet_app/config.py`:

- `CRACKXNET_SEVERITY_MODE`
- `CRACKXNET_SEVERITY_LOW_MAX`
- `CRACKXNET_SEVERITY_MEDIUM_MAX`
- `CRACKXNET_SEVERITY_AREA_WEIGHT`
- `CRACKXNET_SEVERITY_CONFIDENCE_WEIGHT`
- `CRACKXNET_SEVERITY_ELONGATION_WEIGHT`
- `CRACKXNET_EXPLAINABILITY_ENABLED`
- `CRACKXNET_EXPLAINABILITY_METHOD`
- `CRACKXNET_EXPLAINABILITY_TOP_K`

Run the Phase 7 tests:

```bash
python -m pytest tests\test_phase7_explainability_severity.py -q
```

Phase 7 implements explainability and transparent severity reasoning. Phase 8 will implement or refine PASS/REWORK/REJECT quality assessment.

## Phase 8 Quality Assessment

Phase 8 adds a deterministic quality layer after detection, explainability, and severity:

```text
detected defects
  -> LOW / MEDIUM / HIGH severity
  -> rule-based quality assessment
  -> PASS / REWORK / REJECT
```

Default meanings:

- `PASS`: no configured quality issue was found among detections that meet the confidence threshold.
- `REWORK`: one or more defects require correction, but reject rules were not triggered.
- `REJECT`: high-severity defects or excessive defect count triggered a reject rule.

Default priority order:

1. `REJECT` if high-severity defect count meets the configured reject threshold.
2. `REJECT` if total considered defects meet the configured defect-count threshold.
3. `REWORK` if medium-severity defect count meets the configured rework threshold.
4. `REWORK` if accumulated low-severity defects meet the configured rework threshold.
5. `PASS` otherwise.

The quality assessor ignores detections below `CRACKXNET_QUALITY_MIN_CONFIDENCE` for the final quality decision, but reports how many were ignored. This is an engineering rule layer, not a trained quality classifier.

Configuration is centralized through `QualityConfig` in `crackxnet_app/config.py`:

- `CRACKXNET_QUALITY_MODE`
- `CRACKXNET_QUALITY_MIN_CONFIDENCE`
- `CRACKXNET_QUALITY_REJECT_HIGH_COUNT`
- `CRACKXNET_QUALITY_REJECT_DEFECT_COUNT`
- `CRACKXNET_QUALITY_REWORK_MEDIUM_COUNT`
- `CRACKXNET_QUALITY_REWORK_LOW_COUNT`
- `CRACKXNET_QUALITY_PASS_ON_NO_DETECTIONS`

Run the Phase 8 tests:

```bash
python -m pytest tests\test_phase8_quality_assessment.py -q
```

## Troubleshooting

- If `data/DeepPCB` is missing, place or link the local DeepPCB dataset there, or set `DEEPCB_DATA_ROOT`.
- CPU is supported but slow; use `--device cuda` only when CUDA is available.
- If the real checkpoint is missing after clone, run `git lfs pull` and verify `deployment/checkpoints/crackxnet_real_epoch10.pth` exists.
- If a checkpoint is invalid, CLI/API inference exits with a clear error and does not fabricate detections.
- DeepPCB data and generated outputs are ignored by `.gitignore`; do not commit datasets, temporary files, logs, or secrets.

======================================================================

# CrackXNet-PCB

## CrackXNet Hybrid Model Evaluation Results
**Architecture**: EfficientNet-B0 + CBAM + ViT + DDAFF + FPN + Faster R-CNN
**Checkpoint**: `deployment/checkpoints/crackxnet_real_epoch10.pth` (Managed via Git LFS)
**Dataset**: DeepPCB (1000 Train / 500 Test)

### Measured Metrics (Held-out Test Set):
- **mAP@0.50**: 94.87%
- **mAP@0.50:0.95**: 64.90%
- **mAP@0.75**: 75.46%
- **Precision**: 83.09% (at 0.50 Conf / 0.50 IoU)
- **Recall**: 93.73% (at 0.50 Conf / 0.50 IoU)

