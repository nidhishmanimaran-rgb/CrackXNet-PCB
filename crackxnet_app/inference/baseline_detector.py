from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from crackxnet_app.config import DEFAULT_THRESHOLDS, DEFECT_CLASSES, InspectionThresholds
from crackxnet_app.schemas import BoundingBox, DefectPrediction


@dataclass
class Component:
    bbox: BoundingBox
    area: int
    mean_rgb: tuple[float, float, float]
    fill_ratio: float


class BaselinePCBDetector:
    """Classical image-processing baseline for end-to-end MVP behavior.

    This is intentionally not presented as CrackXNet accuracy. It provides useful
    boxes and saliency for demos and integration while trained modules are added.
    """

    def __init__(self, thresholds: InspectionThresholds = DEFAULT_THRESHOLDS) -> None:
        self.thresholds = thresholds

    def detect(self, image: Image.Image) -> tuple[list[DefectPrediction], Image.Image]:
        arr = np.asarray(image.convert("RGB"))
        mask = self._candidate_mask(arr)
        components = self._connected_components(mask, arr)
        defects = [self._classify(component, image.size) for component in components]
        defects.sort(key=lambda item: item.severity, reverse=True)
        heatmap = self._heatmap(mask, image.size)
        return defects, heatmap

    def _candidate_mask(self, arr: np.ndarray) -> np.ndarray:
        rgb = arr.astype(np.float32)
        gray = rgb.mean(axis=2)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        saturation_proxy = np.max(rgb, axis=2) - np.min(rgb, axis=2)

        dark = gray < np.percentile(gray, 10)
        bright = gray > np.percentile(gray, 94)
        copper_like = (r > g * 1.03) & (r > b * 1.08) & (saturation_proxy > 22)
        color_outlier = saturation_proxy > np.percentile(saturation_proxy, 92)

        edges = self._edge_mask(gray)
        mask = (dark | bright | color_outlier | (copper_like & edges))
        return self._morph_close(mask)

    def _edge_mask(self, gray: np.ndarray) -> np.ndarray:
        padded = np.pad(gray, 1, mode="edge")
        gx = padded[1:-1, 2:] - padded[1:-1, :-2]
        gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
        magnitude = np.sqrt(gx * gx + gy * gy)
        return magnitude > np.percentile(magnitude, 88)

    def _morph_close(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        dilated = np.zeros_like(mask, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                dilated |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        padded = np.pad(dilated, 1, mode="constant", constant_values=True)
        eroded = np.ones_like(mask, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return eroded

    def _connected_components(self, mask: np.ndarray, arr: np.ndarray) -> list[Component]:
        height, width = mask.shape
        total_area = height * width
        min_area = max(6, int(total_area * self.thresholds.min_component_area_ratio))
        max_area = int(total_area * self.thresholds.max_component_area_ratio)
        visited = np.zeros_like(mask, dtype=bool)
        components: list[Component] = []

        ys, xs = np.where(mask)
        for start_y, start_x in zip(ys.tolist(), xs.tolist()):
            if visited[start_y, start_x]:
                continue
            pixels: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            area = len(pixels)
            if area < min_area or area > max_area:
                continue
            py = np.array([p[0] for p in pixels])
            px = np.array([p[1] for p in pixels])
            x1, x2 = int(px.min()), int(px.max()) + 1
            y1, y2 = int(py.min()), int(py.max()) + 1
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            fill_ratio = area / max(1, bbox.area)
            colors = arr[py, px, :].mean(axis=0)
            components.append(
                Component(
                    bbox=bbox,
                    area=area,
                    mean_rgb=(float(colors[0]), float(colors[1]), float(colors[2])),
                    fill_ratio=float(fill_ratio),
                )
            )
        return self._merge_nearby(components)

    def _merge_nearby(self, components: list[Component]) -> list[Component]:
        merged: list[Component] = []
        for component in components:
            added = False
            for index, existing in enumerate(merged):
                if self._boxes_near(component.bbox, existing.bbox):
                    bbox = BoundingBox(
                        min(component.bbox.x1, existing.bbox.x1),
                        min(component.bbox.y1, existing.bbox.y1),
                        max(component.bbox.x2, existing.bbox.x2),
                        max(component.bbox.y2, existing.bbox.y2),
                    )
                    area = component.area + existing.area
                    mean_rgb = tuple((a + b) / 2.0 for a, b in zip(component.mean_rgb, existing.mean_rgb))
                    fill_ratio = area / max(1, bbox.area)
                    merged[index] = Component(bbox, area, mean_rgb, fill_ratio)
                    added = True
                    break
            if not added:
                merged.append(component)
        return merged

    def _boxes_near(self, a: BoundingBox, b: BoundingBox) -> bool:
        horizontal_gap = max(0, max(a.x1, b.x1) - min(a.x2, b.x2))
        vertical_gap = max(0, max(a.y1, b.y1) - min(a.y2, b.y2))
        return sqrt(horizontal_gap * horizontal_gap + vertical_gap * vertical_gap) <= 3

    def _classify(self, component: Component, image_size: tuple[int, int]) -> DefectPrediction:
        width, height = image_size
        image_area = width * height
        bbox = component.bbox
        aspect = bbox.width / max(1, bbox.height)
        inverse_aspect = bbox.height / max(1, bbox.width)
        area_ratio = component.area / max(1, image_area)
        r, g, b = component.mean_rgb

        if component.fill_ratio < 0.22 and component.area < image_area * 0.003:
            label = "Pin Hole"
            rationale = "Small sparse bright/dark anomaly consistent with a hole-like defect."
        elif max(aspect, inverse_aspect) > 7:
            label = "Open Circuit" if component.fill_ratio < 0.5 else "Short Circuit"
            rationale = "Long thin anomaly on a trace-like region."
        elif r > g * 1.08 and r > b * 1.12 and component.fill_ratio > 0.45:
            label = "Spurious Copper"
            rationale = "Copper-colored extra material candidate."
        elif component.area < image_area * 0.0015:
            label = "Mouse Bite"
            rationale = "Small edge-like missing material candidate."
        elif component.fill_ratio > 0.65 and max(bbox.width, bbox.height) > 20:
            label = "Spur"
            rationale = "Dense protrusion-like connected anomaly."
        else:
            label = DEFECT_CLASSES[component.area % len(DEFECT_CLASSES)]
            rationale = "Baseline visual anomaly; final class should be verified by a trained classifier."

        severity = min(1.0, 0.18 + area_ratio * 55.0 + min(0.32, max(aspect, inverse_aspect) / 30.0))
        confidence = min(0.93, 0.42 + component.fill_ratio * 0.25 + min(0.25, area_ratio * 40.0))
        return DefectPrediction(
            label=label,
            confidence=round(float(confidence), 3),
            severity=round(float(severity), 3),
            bbox=bbox,
            rationale=rationale,
        )

    def _heatmap(self, mask: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
        gray = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        gray = gray.filter(ImageFilter.GaussianBlur(radius=self.thresholds.heatmap_blur_radius))
        if gray.size != image_size:
            gray = gray.resize(image_size)
        return gray


def draw_overlay(image: Image.Image, defects: list[DefectPrediction]) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for defect in defects:
        bbox = defect.bbox
        color = _severity_color(defect.severity)
        draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=color, width=3)
        text = f"{defect.label} {defect.confidence:.2f}"
        text_box = draw.textbbox((bbox.x1, bbox.y1), text)
        pad = 4
        draw.rectangle(
            [text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad],
            fill=color,
        )
        draw.text((bbox.x1, bbox.y1), text, fill=(255, 255, 255))
    return overlay


def compose_heatmap(image: Image.Image, heatmap: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    heat = heatmap.convert("L").resize(base.size)
    red = Image.new("RGBA", base.size, (255, 38, 38, 0))
    red.putalpha(heat.point(lambda value: int(value * 0.55)))
    return Image.alpha_composite(base, red).convert("RGB")


def _severity_color(severity: float) -> tuple[int, int, int]:
    if severity >= 0.72:
        return (210, 35, 35)
    if severity >= 0.35:
        return (222, 139, 22)
    return (35, 130, 74)
