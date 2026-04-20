"""Image feature analysis for adaptive parameter selection."""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def analyze_image(img_rgb: np.ndarray) -> Dict[str, float]:
    """Analyse *img_rgb* (H×W×3, uint8, RGB) and return feature metrics.

    Returns
    -------
    dict with keys:
        h, w            – image dimensions
        noise           – Laplacian variance (proxy for noise / texture)
        unique_colors   – number of unique colours after 5-bit quantisation
        edge_ratio      – fraction of Canny edge pixels
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # (a) Noise level: Laplacian variance
    noise_level: float = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # (b) Colour complexity: unique colours after 5-bit quantisation
    q = (img_rgb >> 3).astype(np.int32)
    packed = q[..., 0] * 1024 + q[..., 1] * 32 + q[..., 2]
    unique_colors: int = int(np.unique(packed).size)

    # (c) Edge density: Canny edge pixel fraction
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio: float = float(edges.mean() / 255.0)

    return {
        "h": float(h),
        "w": float(w),
        "noise": noise_level,
        "unique_colors": float(unique_colors),
        "edge_ratio": edge_ratio,
    }
