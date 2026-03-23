"""
Heuristic radiographic features for alveolar bone appearance on panoramic-style images.

This does NOT measure clinical bone loss percentage (that needs CEJ–alveolar crest
distances or expert segmentation). It produces a reproducible *proxy index* and
raw cues you can correlate with stages as you collect more data and optional labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from PIL import Image


_SEVERITY_THRESHOLDS: list[tuple[float, int, str, str]] = [
    # (upper_bound, approx_stage, severity_label, plain_description)
    (0.15, 0, "Healthy",        "No significant radiographic bone-loss indicators detected."),
    (0.35, 1, "Mild",           "Minor alveolar changes consistent with early periodontal involvement (Stage I)."),
    (0.55, 2, "Moderate",       "Noticeable alveolar bone changes suggesting moderate periodontitis (Stage II)."),
    (0.75, 3, "Severe",         "Pronounced bone-loss indicators consistent with severe periodontitis (Stage III)."),
    (1.01, 4, "Very severe",    "Extensive radiographic bone-loss pattern consistent with advanced periodontitis (Stage IV)."),
]


@dataclass
class BoneLossFeatures:
    """Numeric summaries from a single radiograph."""

    alveolar_mean: float
    upper_reference_mean: float
    radiolucency_ratio: float
    edge_strength: float
    bone_loss_index: float

    def interpret(self) -> Dict[str, Any]:
        """Map bone_loss_index to a human-readable severity and approximate stage."""
        for upper, stage, label, description in _SEVERITY_THRESHOLDS:
            if self.bone_loss_index < upper:
                return {
                    "severity": label,
                    "approximate_stage": stage,
                    "description": description,
                    "bone_loss_index": round(self.bone_loss_index, 3),
                }
        last = _SEVERITY_THRESHOLDS[-1]
        return {
            "severity": last[2],
            "approximate_stage": last[1],
            "description": last[3],
            "bone_loss_index": round(self.bone_loss_index, 3),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alveolar_mean": self.alveolar_mean,
            "upper_reference_mean": self.upper_reference_mean,
            "radiolucency_ratio": self.radiolucency_ratio,
            "edge_strength": self.edge_strength,
            "bone_loss_index": self.bone_loss_index,
            "interpretation": self.interpret(),
        }


def _to_gray_float(path_or_array: Union[Path, str, np.ndarray]) -> np.ndarray:
    if isinstance(path_or_array, (Path, str)):
        img = Image.open(path_or_array).convert("L")
        gray = np.asarray(img, dtype=np.float32) / 255.0
    else:
        g = path_or_array
        if g.ndim == 3:
            g = np.mean(g, axis=-1)
        gray = g.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0
    return gray


def _band_means(gray: np.ndarray) -> tuple[float, float, float]:
    h, w = gray.shape
    y_upper0, y_upper1 = int(0.10 * h), int(0.32 * h)
    y_alv0, y_alv1 = int(0.38 * h), int(0.92 * h)
    upper = gray[y_upper0:y_upper1, :]
    alveolar = gray[y_alv0:y_alv1, :]
    full_mean = float(np.mean(gray))
    return float(np.mean(upper)), float(np.mean(alveolar)), full_mean


def _edge_strength(gray: np.ndarray) -> float:
    h, w = gray.shape
    y0, y1 = int(0.38 * h), int(0.92 * h)
    band = gray[y0:y1, :]
    if band.shape[0] < 3 or band.shape[1] < 3:
        return 0.0
    gx = np.abs(band[:, 1:] - band[:, :-1])
    gy = np.abs(band[1:, :] - band[:-1, :])
    return float(0.5 * (np.mean(gx) + np.mean(gy)))


def analyze_bone_loss_proxy(
    path_or_gray: Union[Path, str, np.ndarray],
    eps: float = 1e-6,
) -> BoneLossFeatures:
    gray = _to_gray_float(path_or_gray)
    upper_m, alv_m, _full_m = _band_means(gray)
    edges = _edge_strength(gray)
    # More radiolucency in alveolar band vs upper reference → ratio > 1 suggests darker alveolar.
    radiolucency_ratio = (upper_m + eps) / (alv_m + eps)
    # Map to [0, 1]: typical ratios cluster ~0.8–1.4 on mixed panoramics; widen if needed.
    r = float(np.clip((radiolucency_ratio - 0.75) / 0.55, 0.0, 1.0))
    e = float(np.clip((edges - 0.02) / 0.12, 0.0, 1.0))
    index = float(np.clip(0.65 * r + 0.35 * e, 0.0, 1.0))
    return BoneLossFeatures(
        alveolar_mean=alv_m,
        upper_reference_mean=upper_m,
        radiolucency_ratio=radiolucency_ratio,
        edge_strength=edges,
        bone_loss_index=index,
    )


def compare_to_reference(
    patient: Union[Path, str, np.ndarray],
    reference_healthy: Union[Path, str, np.ndarray],
) -> Dict[str, Any]:
    """Delta of proxy index vs a chosen healthy reference image (same scanner if possible)."""
    p = analyze_bone_loss_proxy(patient)
    r = analyze_bone_loss_proxy(reference_healthy)
    return {
        "patient": p.as_dict(),
        "reference": r.as_dict(),
        "delta_bone_loss_index": p.bone_loss_index - r.bone_loss_index,
        "delta_radiolucency_ratio": p.radiolucency_ratio - r.radiolucency_ratio,
    }
