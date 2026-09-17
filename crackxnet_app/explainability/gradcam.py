from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from crackxnet_app.config import DEFAULT_EXPLAINABILITY_CONFIG, ExplainabilityConfig
from crackxnet_app.inference.baseline_detector import compose_heatmap


@dataclass(frozen=True)
class ExplainabilityResult:
    raw_heatmap: torch.Tensor
    normalized_heatmap: Image.Image
    overlay: Image.Image
    method: str
    target_layer: str
    note: str


class GradCAMExplainer:
    """Grad-CAM style detector explainer.

    For the hybrid detector, the default target layer is `model.backbone.fusion.norm`,
    which is directly after DDAFF and before FPN. The target score is the sum of
    the top-k detection confidence scores from the actual detector forward pass.
    """

    def __init__(self, config: ExplainabilityConfig = DEFAULT_EXPLAINABILITY_CONFIG) -> None:
        if config.method != "grad_cam":
            raise ValueError("Only grad_cam explainability is implemented.")
        if config.top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.config = config

    def explain_detector(self, detector, image: Image.Image, defects: list | None = None) -> ExplainabilityResult:
        if not self.config.enabled:
            return self._empty(image, "disabled", "Explainability disabled.")
        if defects is not None and len(defects) == 0:
            return self._empty(image, "no_target", "No detections above threshold; no Grad-CAM target selected.")
        model = getattr(detector, "model", None)
        image_size = int(getattr(detector, "image_size", max(image.size)))
        device = getattr(detector, "torch_device", torch.device("cpu"))
        if model is None:
            return self._empty(image, "not_available", "Detector has no torch model; using no learned explanation.")
        target_layer = self._resolve_target_layer(model)
        if target_layer is None:
            return self._empty(image, "not_available", "No supported Grad-CAM target layer found.")
        return self.explain_model(model, target_layer, image, image_size, device)

    def explain_model(self, model, target_layer, image: Image.Image, image_size: int, device) -> ExplainabilityResult:
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        try:
            model.eval()
            model.zero_grad(set_to_none=True)
            tensor = F.to_tensor(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR))
            tensor = tensor.to(device).requires_grad_(True)
            with torch.enable_grad():
                output = model([tensor])[0]
                scores = output.get("scores")
                if scores is None or scores.numel() == 0:
                    return self._empty(image, self._target_name(model, target_layer), "Model produced no detections.")
                target_score = scores[: self.config.top_k].sum()
                target_score.backward()
            if not activations or not gradients:
                return self._empty(image, self._target_name(model, target_layer), "Target layer did not expose gradients.")
            raw = self._gradcam(activations[-1].detach(), gradients[-1].detach())
            heatmap = self._to_image(raw, image.size)
            return ExplainabilityResult(
                raw_heatmap=raw.cpu(),
                normalized_heatmap=heatmap,
                overlay=compose_heatmap(image, heatmap),
                method="grad_cam",
                target_layer=self._target_name(model, target_layer),
                note="Grad-CAM from actual detector scores.",
            )
        finally:
            handle_fwd.remove()
            handle_bwd.remove()

    def _gradcam(self, activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 4 or gradient.ndim != 4:
            raise ValueError("Grad-CAM requires NCHW activation and gradient tensors.")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=False))[0]
        cam = cam - cam.min()
        max_value = cam.max()
        if float(max_value) > 0:
            cam = cam / max_value
        return cam

    def _to_image(self, heatmap: torch.Tensor, size: tuple[int, int]) -> Image.Image:
        array = (heatmap.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L").resize(size, Image.Resampling.BILINEAR)

    def _empty(self, image: Image.Image, target_layer: str, note: str) -> ExplainabilityResult:
        raw = torch.zeros((1, 1), dtype=torch.float32)
        heatmap = Image.new("L", image.size, 0)
        return ExplainabilityResult(raw, heatmap, compose_heatmap(image, heatmap), "grad_cam", target_layer, note)

    def _resolve_target_layer(self, model):
        backbone = getattr(model, "backbone", None)
        fusion = getattr(backbone, "fusion", None)
        if fusion is not None and hasattr(fusion, "norm"):
            return fusion.norm
        if backbone is not None and hasattr(backbone, "body"):
            return backbone.body
        return None

    def _target_name(self, model, target_layer) -> str:
        for name, module in model.named_modules():
            if module is target_layer:
                return name
        return target_layer.__class__.__name__
